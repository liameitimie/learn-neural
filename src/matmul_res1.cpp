#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

int main(int argc, char *argv[]) {
    Context ctx{argv[0]};
    Device device = ctx.create_device("dx");
    Stream stream = device.create_stream();

    const uint layer_width = 64;
    const uint batch_size = 16384;
    const uint tile_k = 128;
    const uint split_k = batch_size / tile_k;

    vector<float> a_h(layer_width * batch_size);
    vector<float> b_h(layer_width * batch_size);
    vector<float> c_h(layer_width * layer_width);
    vector<float> c_buffer(layer_width * layer_width);

    srand(0);
    for (int i = 0; i < layer_width * batch_size; i++) {
        a_h[i] = rand() / 65536.0;
        b_h[i] = rand() / 65536.0;
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

    auto a = device.create_buffer<float>(layer_width * batch_size);
    auto b = device.create_buffer<float>(layer_width * batch_size);
    auto c = device.create_buffer<float>(layer_width * layer_width);
    auto tmp = device.create_buffer<float>(layer_width * layer_width * split_k);

    stream << a.copy_from(a_h.data()) << b.copy_from(b_h.data()) << synchronize();

    Kernel1D kernel1 = [&]() {
        set_block_size(256);
        $shared<float4> a_tile{64};
        $shared<float4> b_tile{64};
        $float4 a_flag;
        $float4 b_flag;
        $float4 acc[4];
        
        $ lane_idx = $dispatch_x % 32;
        $ warp_idx = $dispatch_x / 32 % 8;
        $ tk = $dispatch_x / 256;

        $ x_ofs = lane_idx % 8 + warp_idx / 4 * 8;
        $ y_ofs = lane_idx / 8 + warp_idx % 4 * 4;

        $for (t, tile_k / 4) {
            $ tid = $dispatch_x % 256;
            $ va = a->read(tid + t*256 + tk*(tile_k*layer_width));
            $ vb = b->read(tid + t*256 + tk*(tile_k*layer_width));
            a_tile[tid / 4][tid % 4] = va;
            b_tile[tid / 4][tid % 4] = vb;

            sync_block();

            for (int t1 = 0; t1 < 4; t1++) {
                a_flag = a_tile[x_ofs + t1 * layer_width / 4];
                b_flag = b_tile[y_ofs + t1 * layer_width / 4];
                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        acc[y][x] += a_flag[x] * b_flag[y];
                    }
                }
            };
            sync_block();
        };
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                tmp->write(
                    (x + x_ofs*4) + (y + y_ofs*4)*layer_width + tk*(layer_width*layer_width),
                    acc[y][x]
                );
            }
        }
    };
    Kernel1D kernel2 = [&]() {
        set_block_size(256);
        $ idx = $dispatch_x;
        $float cc = 0;
        $for (tk, split_k) {
            cc += tmp->read(idx + tk*layer_width*layer_width);
        };
        c->write(idx, cc);
    };

    auto shader1 = device.compile(kernel1);
    auto shader2 = device.compile(kernel2);
    
    for (int i = 0; i < 100; i++) {
        timer.tic();
        for (int i = 0; i < 1000; i++) {
            stream << shader1().dispatch(256 * split_k);
            // stream << shader1().dispatch(32, 8, split_k);
            stream << shader2().dispatch(layer_width * layer_width);
            // stream.synchronize();
        }
        stream.synchronize();
        print("{}\n", timer.toc());
    }

    stream << c.copy_to(c_buffer.data()) << synchronize();

    float f_err = 0;
    for (int i = 0; i < layer_width * layer_width; i++) {
        f_err = max(f_err, abs(c_h[i] - c_buffer[i]));
        // if (c_h[i] != c_buffer[i]) {
        //     print("error {}: {}, {}\n", i, c_h[i], c_buffer[i]);
        //     if (i >= 128) break;
        // }
    }
    print("f_err: {}\n", f_err);
    print("ok\n");

    return 0;
}
