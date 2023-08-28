#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

void wmma_load($uint4 &frag, $buffer<uint4> &buf, $uint offset) {
    frag = buf.read(offset + warp_lane_id());
};
void wmma_load($uint4 &frag, $shared<uint4> &smem, $uint offset) {
    $ lid = warp_lane_id();
    frag = smem.read(offset + lid + lid / 8);
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
};
Callable packed_half_add_single = []($uint a, $uint b) -> $uint {
    $uint t1 = (a.as<half>() + b.as<half>()).as<ushort>();
    $uint t2 = ((a >> 16).as<half>() + (b >> 16).as<half>()).as<ushort>();
    return (t2 << 16) + t1;
};
Callable packed_half_add = []($uint4 a, $uint4 b) -> $uint4 {
    for (int i = 0; i < 4; i++) {
        a[i] = packed_half_add_single(a[i], b[i]);
    }
    return a;
};
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
    const uint batch_size = 1920 * 1080;

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

    auto t_mma = [&](float* c, float* a, float* b) {
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
    // for (int ty = 0; ty < batch_size / 16; ty++) {
    //     for (int tx = 0; tx < layer_width / 16; tx++) {
    //         for (int tk = 0; tk < layer_width / 16; tk++) {
    //             t_mma(c_h.data() + (tx + ty * 4) * 256,
    //                 a_tmp.data() + (tx + tk * 4) * 256,
    //                 b_tmp.data() + (tk + ty * 4) * 256
    //             );
    //         }
    //     }
    // }
    // print("{}\n", timer.toc());
    // print("[");
    // for (int i = 0; i < 32; i++) {
    //     print("{}", c_h[i]);
    //     if (i < 31) print(", ");
    // }
    // print("]\n");

    auto a = device.create_buffer<uint4>(layer_width * layer_width / 8);
    auto b = device.create_buffer<uint4>(layer_width * batch_size / 8);
    auto c = device.create_buffer<uint4>(layer_width * batch_size / 8);

    stream << a.copy_from(a_h.data()) << b.copy_from(b_h.data());

    const uint n_iter = 4;
    const uint n_block = layer_width / 16;
    const uint tile_size = layer_width * n_iter * 16 / 8;
    const uint smem_size = (layer_width / 8 + 1) * n_iter * 16;

    Kernel3D kernel = [&]($buffer<uint4> a, $buffer<uint4> b, $buffer<uint4> c) {
        set_block_size(32, n_block);

        $shared<uint4> b_tile{smem_size};
        $uint4 a_frag[n_block];
        $uint4 b_frag;
        $uint4 c_frag[n_iter];

        $ lid = $dispatch_x;
        $ wid = $dispatch_y;
        $ tid = $dispatch_z;

        for (int i = 0; i < n_iter; i++) {
            b_tile[lid + lid / 8 + wid*36 + i*144] = b.read(lid + wid*32 + i*128 + tid*tile_size);
        }
        sync_block();

        for (int i = 0; i < n_block; i++) {
            wmma_load(a_frag[i], a, (wid+i*4)*32);
        }
        for (int j = 0; j < n_iter; j++) {
            for (int i = 0; i < n_block; i++) {
                wmma_load(b_frag, b_tile, (i + j * 4) * 36);
                wmma(c_frag[j], a_frag[i], b_frag);
            }
        }

        for (int j = 0; j < n_iter; j++) {
            wmma_store(c_frag[j], c, (wid+j*4)*32 + tid*tile_size);
        }
    };

    timer.tic();
    auto shader = device.compile(kernel);
    print("compiled shader: {}\n", timer.toc());

    stream.synchronize();

    for (int i = 0; i < 5; i++) {
        timer.tic();
        for (int i = 0; i < 100; i++) {
            stream << shader(a, b, c).dispatch(32, n_block, batch_size / (n_iter*16));
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