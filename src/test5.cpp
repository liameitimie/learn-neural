#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

// struct Fragment {
//     $uint4 x;
// };

void wmma_load($uint4 &frag, $buffer<uint4> &buf, $uint offset) {
    frag = buf.read(offset + warp_lane_id());
};
void wmma_load($uint4 &frag, $shared<uint4> &smem, $uint offset) {
    frag = smem.read(offset + warp_lane_id());
};
void wmma_store($uint4 &frag, $buffer<uint4> &buf, $uint offset) {
    buf.write(offset + warp_lane_id(), frag);
};

Callable packed_half_dot_single = []($uint a, $uint b) -> $half {
    return a.as<half>() * b.as<half>()
        + (a >> 16).as<half>() * (b >> 16).as<half>();
};
Callable packed_half_dot = []($uint4 a, $uint4 b) -> $half {
    $half res = 0;
    for (int i = 0; i < 4; i++) {
        res += packed_half_dot_single(a[i], b[i]);
    }
    return res;
    // return packed_half_dot_single(a.x, b.x)
    //     + packed_half_dot_single(a.y, b.y)
    //     + packed_half_dot_single(a.z, b.z)
    //     + packed_half_dot_single(a.w, b.w);
};
// Callable packed_half_dot_single = []($uint a, $uint b) -> $half {
//     return a.as<half>() * b.as<half>()
//         + (a >> 16).as<half>() * (b >> 16).as<half>();
// };
// Callable packed_half_dot = []($uint4 a, $uint4 b) -> $half {
//     // $half res = 0;
//     // for (int i = 0; i < 4; i++) {
//     //     res += packed_half_dot_single(a[i], b[i]);
//     // }
//     // return res;
//     return packed_half_dot_single(a.x, b.x)
//         + packed_half_dot_single(a.y, b.y)
//         + packed_half_dot_single(a.z, b.z)
//         + packed_half_dot_single(a.w, b.w);
// };
Callable packed_half_add_index_single = []($uint x, $uint idx, $half v) -> $uint {
    // idx is 0 or 1
    $uint2 tx = {x & 0xffff, x >> 16};
    tx[idx] = (tx[idx].as<half>() + v).as<ushort>();
    return (tx[1] << 16) + tx[0];
};
void packed_half_add_index($uint4 &x, $uint idx, $half v) {
    x[idx >> 1] = packed_half_add_index_single(x[idx >> 1], idx & 1, v);
}
Callable wmma_impl = []($uint4 c_frag, $uint4 a_frag, $uint4 b_frag) -> $uint4 {
    $ lid = warp_lane_id();
    $half tmp0;
    $half tmp1;
    $uint op = ((lid >> 3) ^ (lid >> 4)) & 1;

    tmp0 = packed_half_dot(a_frag, b_frag);
    tmp1 = warp_read_lane(tmp0, lid ^ 16);
    packed_half_add_index(c_frag, lid & 7, (op^1) * (tmp0 + tmp1));

    for (int t = 0; t < 2; t++) {
        for (int ofs = 1 - t; ofs < 8; ofs++) {
            $ t_frag = warp_read_lane(b_frag, lid ^ (ofs ^ (8*t)));
            tmp0 = packed_half_dot(a_frag, t_frag);
            tmp1 = warp_read_lane(tmp0, lid ^ 16);
            packed_half_add_index(c_frag, (lid ^ ofs) & 7, (op^(1-t)) * (tmp0 + tmp1));
        }
    }
    return c_frag;
};
void wmma($uint4 &c_frag, $uint4 &a_frag, $uint4 &b_frag) {
    c_frag = wmma_impl(c_frag, a_frag, b_frag);
}


int main(int argc, char *argv[]) {
    Context ctx{argv[0]};
    Device device = ctx.create_device("dx");
    Stream stream = device.create_stream();

    const uint layer_width = 64;
    const uint batch_size = 16384;
    const uint tile_k = 64;
    const uint split_k = batch_size / tile_k;

    vector<half> a_h(layer_width * batch_size);
    vector<half> b_h(layer_width * batch_size);
    vector<float> c_h(layer_width * layer_width);
    vector<half> c_buffer(layer_width * layer_width);

    srand(0);
    for (int i = 0; i < layer_width * batch_size; i++) {
        a_h[i] = rand() / 65536.0;
        b_h[i] = rand() / 65536.0;
    }

    Clock timer;

    timer.tic();
    // for (int k = 0; k < batch_size; k++) {
    //     for (int y = 0; y < layer_width; y++) {
    //         for (int x = 0; x < layer_width; x++) {
    //             c_h[x + y*layer_width] += a_h[x + k*layer_width] * b_h[y + k*layer_width];
    //         }
    //     }
    // }
    auto t_mma = [&](float* c, half* a, half* b) {
        for (int col = 0; col < 16; col++) {
            for (int row = 0; row < 16; row++) {
                int c_idx = col % 8 + row * 8 + col / 8 * 128;
                for (int i = 0; i < 8; i++) {
                    c[c_idx] += a[i + row * 8] * b[i + col * 8];
                }
                for (int i = 0; i < 8; i++) {
                    c[c_idx] += a[i + row * 8 + 128] * b[i + col * 8 + 128];
                }
            }
        }
    };
    for (int tk = 0; tk < batch_size / 16; tk++) {
        for (int ty = 0; ty < layer_width / 16; ty++) {
            for (int tx = 0; tx < layer_width / 16; tx++) {
                t_mma(c_h.data() + (tx + ty * 4) * 256,
                    a_h.data() + (tx + tk * 4) * 256,
                    b_h.data() + (ty + tk * 4) * 256
                );
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

    auto a = device.create_buffer<uint4>(layer_width * batch_size / 8);
    auto b = device.create_buffer<uint4>(layer_width * batch_size / 8);
    auto c = device.create_buffer<uint4>(layer_width * layer_width / 8);
    auto tmp = device.create_buffer<uint4>(layer_width * layer_width / 8);

    stream << a.copy_from(a_h.data()) << b.copy_from(b_h.data()) << synchronize();

    Kernel3D kernel1 = [&]($buffer<uint4> a, $buffer<uint4> b, $buffer<uint4> tmp) {
        set_block_size(32, layer_width / 16);

        $shared<uint4> b_tile{layer_width * tile_k / 8};
        $uint4 a_frag[tile_k / 16];
        $uint4 b_frag;
        $uint4 c_frag[layer_width / 16];

        $ lid = $dispatch_x;
        $ wid = $dispatch_y;
        $ tk = $dispatch_z;

        for (int i = 0; i < tile_k / 16; i++) {
            b_tile[lid + wid*32 + i*128] = b.read(lid + wid*32 + i*128 + tk*tile_k*layer_width/8);
        }
        sync_block();

        for (int i = 0; i < tile_k / 16; i++) {
            wmma_load(a_frag[i], a, (wid+i*4)*32 + tk*tile_k*layer_width/8);
        }
        for (int j = 0; j < layer_width / 16; j++) {
            for (int i = 0; i < tile_k / 16; i++) {
                wmma_load(b_frag, b_tile, (j + i * 4) * 32);
                wmma(c_frag[j], a_frag[i], b_frag);
            }
        }

        for (int j = 0; j < layer_width / 16; j++) {
            wmma_store(c_frag[j], tmp, (wid+j*4)*32 + tk*layer_width*layer_width/8);
        }
    };
    Kernel1D kernel2 = [&]() {
        $ idx = $dispatch_x;
        // $half cc = 0;
        // $for (tk, split_k) {
        //     cc += tmp->read<half>((idx + tk*layer_width*layer_width) * uint(sizeof(half)));
        // };
        // c->write(idx * uint(sizeof(half)), cc);
    };

    timer.tic();
    auto shader1 = device.compile(kernel1);
    auto shader2 = device.compile(kernel2);
    print("compiled shader: {}\n", timer.toc());

    for (int i = 0; i < 10; i++) {
        timer.tic();
        for (int i = 0; i < 1000; i++) {
            stream << shader1(a, b, tmp).dispatch(32, layer_width / 16, split_k);
            // stream << shader2().dispatch(layer_width * layer_width / 8);
            // stream.synchronize();
        }
        stream.synchronize();
        print("{}\n", timer.toc());
    }

    // stream << c.copy_to(c_buffer.data()) << synchronize();

    // float f_err = 0;
    // for (int i = 0; i < layer_width * layer_width; i++) {
    //     f_err = max(f_err, abs(c_h[i] - c_buffer[i]));
    // }
    // print("f_err: {}\n", f_err);
    // print("ok\n");

    return 0;
}