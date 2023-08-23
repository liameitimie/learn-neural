#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;


struct Fragment {
    $array<uint, 8> x;
};

void wmma_load(Fragment &mat, $buffer<uint> &buf, $uint offset, $uint stride) {
    $ lid = $dispatch_x & 31;
    $for (i, 8) {
        mat.x[i] = buf.read(offset + (lid & 15) + (i + lid / 16 * 8) * stride);
    };
}
void wmma_load(Fragment &mat, $shared<uint> &buf, $uint offset, $uint stride) {
    $ lid = $dispatch_x & 31;
    $for (i, 8) {
        mat.x[i] = buf.read(offset + (lid & 15) + (i + lid / 16 * 8) * stride);
    };
}
void wmma_store(Fragment &mat, $buffer<uint> &buf, $uint offset, $uint stride) {
    $ lid = $dispatch_x & 31;
    $for (i, 8) {
        buf.write(offset + (lid & 15) + (i + lid / 16 * 8) * stride, mat.x[i]);
    };
}
void wmma_store(Fragment &mat, $shared<uint> &buf, $uint offset, $uint stride) {
    $ lid = $dispatch_x & 31;
    $for (i, 8) {
        buf.write(offset + (lid & 15) + (i + lid / 16 * 8) * stride, mat.x[i]);
    };
}
Callable wmma_impl = []($array<uint, 8> &c_frag, $array<uint, 8> &a_frag, $array<uint, 8> &b_frag) {
    $ lid = $dispatch_x & 31;
    $uint tmp0 = 0;
    $uint tmp1;
    $uint op = ((lid >> 3) ^ (lid >> 4)) & 1;

    for (int i = 0; i < 8; i++) tmp0 += a_frag[i] * b_frag[i];
    tmp1 = warp_read_lane(tmp0, lid ^ 16);
    c_frag[lid & 7] += (op ^ 1) * (tmp0 + tmp1);

    for (int t = 0; t < 2; t++) {
        for (int ofs = 1 - t; ofs < 8; ofs++) {
            tmp0 = 0;
            $for (i, 8) {
                tmp1 = warp_read_lane(b_frag[i], lid ^ (ofs ^ (8 * t)));
                tmp0 += a_frag[i] * tmp1;
            };
            tmp1 = warp_read_lane(tmp0, lid ^ 16);
            c_frag[(lid ^ ofs) & 7] += (op ^ (1 - t)) * (tmp0 + tmp1);
        }
    }
};
void wmma(Fragment& c_frag, Fragment& a_frag, Fragment& b_frag) {
    wmma_impl(c_frag.x, a_frag.x, b_frag.x);
}

int main(int argc, char *argv[]) {
    Context ctx{argv[0]};
    Device device = ctx.create_device("dx");
    Stream stream = device.create_stream();

    const uint layer_width = 64;
    const uint batch_size = 16384;
    const uint tile_k = 64;
    const uint split_k = batch_size / tile_k;

    vector<uint> a_h(layer_width * batch_size);
    vector<uint> b_h(layer_width * batch_size);
    vector<uint> c_h(layer_width * layer_width);
    vector<uint> c_buffer(layer_width * layer_width);

    srand(0);
    for (int i = 0; i < layer_width * batch_size; i++) {
        a_h[i] = rand() % 3;
        b_h[i] = rand() % 3;
    }

    Clock timer;

    timer.tic();
    for (int k = 0; k < batch_size; k++) {
        for (int y = 0; y < layer_width; y++) {
            for (int x = 0; x < layer_width; x++) {
                c_h[x + y*layer_width] += a_h[x + k*layer_width] * b_h[y + k*layer_width];
            }
        }
    }
    print("{}\n", timer.toc());
    print("[");
    for (int i = 0; i < 32; i++) {
        print("{}", c_h[i]);
        if (i < 31) print(", ");
    }
    print("]\n");

    auto a = device.create_buffer<uint>(layer_width * batch_size);
    auto b = device.create_buffer<uint>(layer_width * batch_size);
    auto c = device.create_buffer<uint>(layer_width * layer_width);
    auto tmp = device.create_buffer<uint>(layer_width * layer_width * split_k);

    stream << a.copy_from(a_h.data()) << b.copy_from(b_h.data()) << synchronize();

    Kernel3D kernel1 = [&]($buffer<uint> a, $buffer<uint> b, $buffer<uint> tmp) {
        set_block_size(32, layer_width / 16);

        $shared<uint> b_tile{layer_width * tile_k};
        Fragment a_frag[tile_k / 16];
        Fragment b_frag;
        Fragment c_frag[layer_width / 16];

        $ lid = $dispatch_x;
        $ wid = $dispatch_y;
        $ tk = $dispatch_z;

        for (int i = 0; i < tile_k / 2; i++) {
            b_tile[lid + wid * 32 + i * 128] = b.read(lid + wid * 32 + i * 128 + tk*tile_k*layer_width);
        }
        sync_block();

        for (int i = 0; i < tile_k / 16; i++) {
            wmma_load(a_frag[i], a, wid*16 + (i*16 + tk*tile_k)*layer_width, layer_width);
        }
        for (int j = 0; j < layer_width / 16; j++) {
            for (int i = 0; i < tile_k / 16; i++) {
                wmma_load(b_frag, b_tile, j*16 + i*16*layer_width, layer_width);
                wmma(c_frag[j], a_frag[i], b_frag);
            }
        }
        
        for (int j = 0; j < layer_width / 16; j++) {
            wmma_store(
                c_frag[j], 
                tmp, 
                wid*16 + j*16*layer_width + tk*layer_width*layer_width,
                layer_width
            );
        }
    };
    Kernel1D kernel2 = [&]() {
        set_block_size(256);
        $ idx = $dispatch_x;
        $uint cc = 0;
        $for (tk, split_k) {
            cc += tmp->read(idx + tk*layer_width*layer_width);
        };
        c->write(idx, cc);
    };

    timer.tic();
    auto shader1 = device.compile(kernel1);
    auto shader2 = device.compile(kernel2);
    print("compiled shader: {}\n", timer.toc());
    
    for (int i = 0; i < 10; i++) {
        timer.tic();
        for (int i = 0; i < 1000; i++) {
            stream << shader1(a, b, tmp).dispatch(32, layer_width / 16, split_k);
        }
        stream.synchronize();
        print("{}\n", timer.toc());
    }

    for (int i = 0; i < 10; i++) {
        timer.tic();
        for (int i = 0; i < 1000; i++) {
            stream << shader2().dispatch(layer_width * layer_width);
        }
        stream.synchronize();
        print("{}\n", timer.toc());
    }

    for (int i = 0; i < 10; i++) {
        timer.tic();
        for (int i = 0; i < 1000; i++) {
            stream << shader1(a, b, tmp).dispatch(32, layer_width / 16, split_k);
            stream << shader2().dispatch(layer_width * layer_width);
            // stream.synchronize();
        }
        stream.synchronize();
        print("{}\n", timer.toc());
    }

    stream << c.copy_to(c_buffer.data()) << synchronize();

    int err_t = 0;
    for (int i = 0; i < layer_width * layer_width; i++) {
        if (c_h[i] != c_buffer[i]) {
            print("error {}: {}, {}\n", i, c_h[i], c_buffer[i]);
            err_t++;
            if (err_t >= 32) break;
        }
    }
    print("ok\n");

    return 0;
}
