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

    vector<uint> a_h(layer_width * batch_size);
    vector<uint> b_h(layer_width * batch_size);
    vector<uint> c_h(layer_width * layer_width);
    vector<uint> c_buffer(layer_width * layer_width);

    srand(0);
    for (int i = 0; i < layer_width * batch_size; i++) {
        a_h[i] = rand() % 4;
        b_h[i] = rand() % 4;
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

    Kernel3D kernel1 = [&]() {
        $ x = $dispatch_x;
        $ y = $dispatch_y;
        $ tk = $dispatch_z;

        $uint cc = 0;
        $for (i, tile_k) {
            $ k = i + tk * tile_k;
            $ t1 = a->read(x + k * layer_width);
            $ t2 = b->read(y + k * layer_width);

            cc += t1 * t2;
        };
        tmp->write(x + y*layer_width + tk*layer_width*layer_width, cc);
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

    Kernel3D kernel3 = [&]() {
        set_block_size(32, 8);
        // $shared<uint> a_tile{256};
        // $shared<uint> b_tile{256};
        $shared<uint4> a_tile{64};
        $shared<uint4> b_tile{64};
        $uint4 a_flag;
        $uint4 b_flag;
        // $array<uint, 16> acc;
        $uint4 acc[4];

        $ lane_idx = $dispatch_x;
        $ warp_idx = $dispatch_y;
        $ tk = $dispatch_z;

        $ x_ofs = (lane_idx % 8) * 4 + (warp_idx / 4) * 32;
        $ y_ofs = (lane_idx / 8) * 4 + (warp_idx % 4) * 16;
        // $ x_ofs = lane_idx % 8 + warp_idx / 4 * 8;
        // $ y_ofs = lane_idx / 8 + warp_idx % 4 * 4;

        $for (t, tile_k / 4) {
            $ tid = lane_idx + warp_idx * 32;
            $ va = a->read(tid + t*256 + tk*(tile_k*layer_width));
            $ vb = b->read(tid + t*256 + tk*(tile_k*layer_width));
            a_tile[tid / 4][tid % 4] = va;
            b_tile[tid / 4][tid % 4] = vb;

            // $ k = t * 4 + tk * tile_k;
            // $ fetch_x = lane_idx + (warp_idx % 2) * 32;
            // $ fetch_k = k + warp_idx / 2;
            // $ store_idx = fetch_x + (warp_idx / 2) * layer_width;
            // $ va = a->read(fetch_x + fetch_k * layer_width);
            // $ vb = b->read(fetch_x + fetch_k * layer_width);
            // a_tile[store_idx] = va;
            // b_tile[store_idx] = vb;
            // a_tile[store_idx / 4][store_idx % 4] = va;
            // b_tile[store_idx / 4][store_idx % 4] = vb;
            
            sync_block();

            for (int t1 = 0; t1 < 4; t1++) {
            // $for (t1, 4) {
                // for (int i = 0; i < 4; i++) {
                //     a_flag[i] = a_tile[i + x_ofs + t1 * layer_width];
                //     b_flag[i] = b_tile[i + y_ofs + t1 * layer_width];
                // }
                a_flag = a_tile[(x_ofs + t1 * layer_width) / 4];
                b_flag = b_tile[(y_ofs + t1 * layer_width) / 4];
                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        // acc[x + y * 4] += a_flag[x] * b_flag[y];
                        acc[y][x] += a_flag[x] * b_flag[y];
                    }
                }
            };
            sync_block();
        };
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                tmp->write(
                    (x + x_ofs) + (y + y_ofs)*layer_width + tk*(layer_width*layer_width),
                    acc[y][x]
                );
            }
        }
    };

    const uint calc_pixels = 16;
    Kernel2D kernel4 = [&]() {
        set_block_size(256);
        $shared<uint> acc{1};
        $ idx = $dispatch_x;
        $ tp = $dispatch_y;

        $if (idx == 0) { acc[0] = 0; };
        sync_block();

        for (int i = 0; i < calc_pixels; i++) {
            $ pid = i + tp * calc_pixels;
            for (int j = 0; j < split_k; j += 256) {
                acc.atomic(0).fetch_add(tmp->read(idx + j + pid * split_k));
            }
            sync_block();
            $if (idx == 0) {
                c->write(pid, acc[0]);
                acc[0] = 0;
            };
        }
        // $uint cc = 0;
        // $for (tk, split_k) {
        //     cc += tmp->read(idx + tk*layer_width*layer_width);
        // };
        // c->write(idx, cc);
    };

    Kernel3D kernel5 = [&]() {
        set_block_size(32, 2);
        // $shared<uint> a_tile{layer_width};
        // $shared<uint> b_tile{layer_width};
        $shared<uint4> a_tile{layer_width / 4};
        $shared<uint4> b_tile{layer_width / 4};
        $uint4 a_flag[2];
        $uint4 b_flag[2];
        $array<uint, 64> acc;

        $ lane_idx = $dispatch_x;
        $ warp_idx = $dispatch_y;
        $ tk = $dispatch_z;

        $ x_ofs = (lane_idx % 8);
        $ y_ofs = (lane_idx / 8) + warp_idx * 8;

        $for (t, tile_k) {
            $ k = t + tk * tile_k;
            $ fetch_x = lane_idx + warp_idx * 32;
            $ va = a->read(fetch_x + k * layer_width);
            $ vb = b->read(fetch_x + k * layer_width);
            a_tile[fetch_x / 4][fetch_x % 4] = va;
            b_tile[fetch_x / 4][fetch_x % 4] = vb;

            sync_block();

            for (int i = 0; i < 4; i++) {
                // if (i % 2 == 0) {
                //     for (int j = 0; j < 4; j++)
                //         a_flag[i / 2][j] = a_tile[j + (x_ofs + i / 2 * 8) * 4];
                // }
                // if (i / 2 == 0) {
                //     for (int j = 0; j < 4; j++)
                //         b_flag[i % 2][j] = b_tile[j + (y_ofs + i % 2 * 4) * 4];
                // }
                if (i % 2 == 0) a_flag[i / 2] = a_tile[x_ofs + i / 2 * 8];
                if (i / 2 == 0) b_flag[i % 2] = b_tile[y_ofs + i % 2 * 4];
                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        acc[x + y * 4 + i * 16] += a_flag[i / 2][x] * b_flag[i % 2][y];
                    }
                }
            }
            sync_block();
        };
        $for (i, 4) {
        // for (int i = 0; i < 4; i++) {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    $ px = x + (x_ofs + i / 2 * 8) * 4;
                    $ py = y + (y_ofs + i % 2 * 4) * 4;
                    tmp->write(
                        px + py*layer_width + tk*(layer_width*layer_width),
                        acc[x + y * 4 + i * 16]
                    );
                }
            }
        };
    };

    auto shader1 = device.compile(kernel1);
    auto shader2 = device.compile(kernel2);
    auto shader3 = device.compile(kernel3);
    auto shader4 = device.compile(kernel4);
    auto shader5 = device.compile(kernel5);

    // timer.tic();
    // for (int i = 0; i < 10000; i++) {
    //     // stream << shader1().dispatch(layer_width, layer_width, split_k);
    //     // stream << shader3().dispatch(256 * split_k);
    //     stream << shader3().dispatch(32, 8, split_k);
    //     // stream << shader5().dispatch(32, 2, split_k);
    //     // stream.synchronize();
    // }
    // stream.synchronize();
    // print("{}\n", timer.toc());

    // timer.tic();
    // for (int i = 0; i < 10000; i++) {
    //     stream << shader2().dispatch(layer_width * layer_width);
    //     // stream << shader4().dispatch(256, layer_width * layer_width / calc_pixels);
    //     // stream.synchronize();
    // }
    // stream.synchronize();
    // print("{}\n", timer.toc());

    timer.tic();
    for (int i = 0; i < 10000; i++) {
        // stream << shader1().dispatch(layer_width, layer_width, split_k);
        // stream << shader3().dispatch(256 * split_k);
        stream << shader3().dispatch(32, 8, split_k);
        // stream << shader5().dispatch(32, 2, split_k);

        stream << shader2().dispatch(layer_width * layer_width);
        // stream << shader4().dispatch(256, layer_width * layer_width / calc_pixels);
        // stream.synchronize();
    }
    stream.synchronize();

    print("{}\n", timer.toc());
    

    stream << c.copy_to(c_buffer.data()) << synchronize();

    for (int i = 0; i < layer_width * layer_width; i++) {
        if (c_h[i] != c_buffer[i]) {
            print("error {}: {}, {}\n", i, c_h[i], c_buffer[i]);
            if (i >= 32) break;
        }
    }
    print("ok\n");

    return 0;
}
