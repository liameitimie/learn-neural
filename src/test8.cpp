#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

int main(int argc, char *argv[]) {
    Context ctx{argv[0]};
    Device device = ctx.create_device("dx");
    Stream stream = device.create_stream();

    const uint layer_width = 64;
    const uint batch_size = 1920 * 1080;
    const uint tile_width = 128;

    vector<half> a_h(layer_width * layer_width);
    vector<half> b_h(layer_width * batch_size);
    vector<float> a_tmp(layer_width * layer_width);
    vector<float> b_tmp(layer_width * batch_size);
    vector<float> c_h(layer_width * batch_size);
    vector<half> c_buffer(layer_width * batch_size);

    srand(0);
    for (int i = 0; i < layer_width * layer_width; i++) {
        a_h[i] = a_tmp[i] = rand() / 65536.0;
    }
    for (int i = 0; i < layer_width * batch_size; i++) {
        b_h[i] = b_tmp[i] = rand() / 65536.0;
    }

    Clock timer;

    timer.tic();
    for (int t = 0; t < batch_size / tile_width; t++) {
        int t_ofs = t*tile_width*layer_width;
        for (int k = 0; k < layer_width; k++) {
            for (int x = 0; x < layer_width; x++) {
                float tmp = a_h[x + k*layer_width];
                for (int y = 0; y < tile_width; y++) {
                    c_h[y + x*tile_width + t_ofs] += tmp * b_h[y + k*tile_width + t_ofs];
                }
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

    auto a = device.create_buffer<uint4>(layer_width * layer_width / 8);
    auto b = device.create_buffer<uint4>(layer_width * batch_size / 8);
    auto c = device.create_buffer<uint4>(layer_width * batch_size / 8);

    stream << a.copy_from(a_h.data()) << b.copy_from(b_h.data()) << synchronize();

    const uint block_size = 128;
    const uint k1_iter = layer_width * layer_width / 8 / block_size;
    const uint k2_iter = tile_width / layer_width;
    const uint k_iter = block_size / (tile_width / 8);
    const uint tile_size = tile_width * layer_width / 8;

    Callable packed_half_fma = []($uint4 a, $half x, $uint4 b) -> $uint4 {
        // return a*x + b
        for (int i = 0; i < 4; i++) {
            $uint t1 = (a[i].as<half>() * x + b[i].as<half>()).as<ushort>();
            $uint t2 = ((a[i] >> 16).as<half>() * x + (b[i] >> 16).as<half>()).as<ushort>();
            a[i] = (t2 << 16) + t1;
        }
        return a;
    };
    auto inner_mma = [&]($uint4 &a_frag, $uint4 &b_frag, $uint4 *acc) {
        for (int i = 0; i < 4; i++) {
            $half t1 = a_frag[i].as<half>();
            acc[i * 2] = packed_half_fma(b_frag, t1, acc[i * 2]);
            $half t2 = (a_frag[i] >> 16).as<half>();
            acc[i * 2 + 1] = packed_half_fma(b_frag, t2, acc[i * 2 + 1]);
        }
    };
    Kernel1D kernel = [&]() {
        set_block_size(block_size);
        $shared<uint4> a_tile{block_size};
        $shared<uint4> b_tile{block_size};
        $uint4 a_frag;
        $uint4 b_frag;
        $uint4 acc[8];

        $ lid = $dispatch_x % 32;
        $ wid = $dispatch_x % block_size / 32;
        $ tid = $dispatch_x % block_size;
        $ bid = $dispatch_x / block_size;

        $ x_ofs = lid % 8;
        $ y_ofs = lid / 8 + wid * 4;

        $for(k1, k1_iter) {
            a_tile[tid] = a->read(tid + k1*block_size);
            $for(k2, k2_iter) {
                b_tile[tid] = b->read(tid + (k2 + k1*k2_iter)*block_size + bid*tile_size);
                sync_block();

                $for(k, k_iter) {
                    a_frag = a_tile[x_ofs + (k + k2*k_iter)*(layer_width/8)];
                    b_frag = b_tile[y_ofs + k*(tile_width/8)];
                    inner_mma(a_frag, b_frag, acc);
                };
                sync_block();
            };
        };
        for (int i = 0; i < 8; i++) {
            c->write(
                y_ofs + (i+x_ofs*8)*tile_width/8 + bid*tile_size, 
                acc[i]
            );
        }
    };

    timer.tic();
    auto shader = device.compile(kernel);
    print("compiled shader: {}\n", timer.toc());

    for (int i = 0; i < 5; i++) {
        timer.tic();
        for (int i = 0; i < 100; i++) {
            stream << shader().dispatch(batch_size);
        }
        stream.synchronize();
        print("{}\n", timer.toc());
    }

    stream << c.copy_to(c_buffer.data()) << synchronize();

    float f_err = 0;
    for (int i = 0; i < layer_width * batch_size; i++) {
        f_err = max(f_err, abs(c_h[i] - c_buffer[i]));
    }
    print("f_err: {}\n", f_err);
    print("ok\n");
    
    return 0;
}