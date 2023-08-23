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
    const uint tile_k = 256;
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

    auto a = device.create_buffer<half4>(layer_width * batch_size / 4);
    auto b = device.create_buffer<half4>(layer_width * batch_size / 4);
    auto c = device.create_buffer<half4>(layer_width * layer_width / 4);
    auto tmp = device.create_buffer<half4>(layer_width * layer_width * split_k / 4);

    stream << a.copy_from(a_h.data()) << b.copy_from(b_h.data()) << synchronize();

    Kernel1D kernel1 = [&]() {
        set_block_size(256);
        // set_block_size(32, 8);
        $shared<half4> a_tile{256};
        $shared<half4> b_tile{256};
        $half4 a_flag;
        $half4 b_flag;
        $half4 acc[4];

        $ lane_idx = $dispatch_x % 32;
        $ warp_idx = $dispatch_x / 32 % 8;
        $ tk = $dispatch_x / 256;
        // $ lane_idx = $dispatch_x;
        // $ warp_idx = $dispatch_y;
        // $ tk = $dispatch_z;

        $ x_ofs = lane_idx % 8 + warp_idx / 4 * 8;
        $ y_ofs = lane_idx / 8 + warp_idx % 4 * 4;

        $for (t, tile_k / 16) {
            $ tid = $dispatch_x % 256;
            // $ tid = lane_idx + warp_idx * 32;
            $ t1 = a->read(tid + t * 256 + tk * (tile_k * layer_width / 4));
            $ t2 = b->read(tid + t * 256 + tk * (tile_k * layer_width / 4));
            a_tile[tid] = t1;
            b_tile[tid] = t2;

            sync_block();

            $for (s, 4) {
                for (int s1 = 0; s1 < 4; s1++) {
                    a_flag = a_tile[x_ofs + (s1 + s * 4) * (layer_width / 4)];
                    b_flag = b_tile[y_ofs + (s1 + s * 4) * (layer_width / 4)];
                    for (int y = 0; y < 4; y++) {
                        for (int x = 0; x < 4; x++) {
                            acc[y][x] += a_flag[x] * b_flag[y];
                        }
                    }
                }
            };
            sync_block();
        };
        for (int y = 0; y < 4; y++) {
            // $half4 t = acc[y];
            tmp->write(
                x_ofs + (y + y_ofs*4)*(layer_width/4) + tk*(layer_width*layer_width/4),
                acc[y]
            );
        }
    };
    Kernel1D kernel2 = [&]() {
        set_block_size(256);
        $ idx = $dispatch_x;
        $float4 cc;
        $for (tk, split_k) {
            $float4 t = tmp->read(idx + tk*(layer_width*layer_width/4));
            cc += t;
        };
        $half4 tcc = cc;
        c->write(idx, tcc);
    };

    auto shader1 = device.compile(kernel1);
    auto shader2 = device.compile(kernel2);
    
    // timer.tic();
    // for (int i = 0; i < 10000; i++) {
    //     stream << shader1().dispatch(256 * split_k);
    //     // stream << shader1().dispatch(32, 8, split_k);
    //     // stream.synchronize();
    // }
    // stream.synchronize();
    // print("{}\n", timer.toc());

    // timer.tic();
    // for (int i = 0; i < 10000; i++) {
    //     stream << shader2().dispatch(layer_width * layer_width / 4);
    //     // stream.synchronize();
    // }
    // stream.synchronize();
    // print("{}\n", timer.toc());

    for (int i = 0; i < 100; i++) {
        timer.tic();
        for (int i = 0; i < 1000; i++) {
            stream << shader1().dispatch(256 * split_k);
            // stream << shader1().dispatch(32, 8, split_k);
            stream << shader2().dispatch(layer_width * layer_width / 4);
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
        //     if (i >= 32) break;
        // }
    }
    print("f_err: {}\n", f_err);
    print("ok\n");

    return 0;
}
