#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

int main(int argc, char *argv[]) {
    Context ctx{argv[0]};
    Device device = ctx.create_device("dx");
    Stream stream = device.create_stream();

    const uint layer_width = 32;
    const uint batch_size = 1920 * 1080;

    vector<half> a_h(layer_width * layer_width);
    vector<half> a1_h(layer_width * layer_width);
    vector<half> b_h(layer_width * batch_size);
    vector<float> a_tmp(layer_width * layer_width);
    vector<float> a1_tmp(layer_width * layer_width);
    vector<float> b_tmp(layer_width * batch_size);
    vector<float> c_h(layer_width * batch_size);
    vector<float> c1_h(layer_width * batch_size);
    vector<half> c_buffer(layer_width * batch_size);

    srand(0);
    for (int i = 0; i < layer_width * layer_width; i++) {
        a_h[i] = a_tmp[i] = rand() / 65536.0;
        a1_h[i] = a1_tmp[i] = rand() / 65536.0;
    }
    for (int i = 0; i < layer_width * batch_size; i++) {
        b_h[i] = b_tmp[i] = rand() / 65536.0;
    }

    Clock timer;

    auto a = device.create_buffer<half4>(layer_width * layer_width / 4);
    auto a1 = device.create_buffer<half4>(layer_width * layer_width / 4);
    auto b = device.create_buffer<half4>(layer_width * batch_size / 4);
    auto c = device.create_buffer<half4>(layer_width * batch_size / 4);

    auto test_mma = [](float* a, float* b, float* c) {
        for (int y = 0; y < batch_size; y++) {
            for (int x = 0; x < layer_width; x++) {
                float s = 0;
                for (int k = 0; k < layer_width; k++) {
                    s += a[k + x * layer_width] * b[k + y * layer_width];
                }
                c[x + y*layer_width] += s;
            }
        }
    };

    timer.tic();

    test_mma(a_tmp.data(), b_tmp.data(), c_h.data());
    // test_mma(a1_tmp.data(), c_h.data(), c1_h.data());
    
    print("{}\n", timer.toc());
    print("[");
    for (int i = 0; i < 32; i++) {
        print("{}", c_h[i]);
        if (i < 31) print(", ");
    }
    print("]\n");

    const uint block_size = 256;
    const uint row_size = layer_width / 4;

    Kernel1D kernel = [&]($buffer<half4> a, $buffer<half4> b, $buffer<half4> c) {
        set_block_size(block_size);
        $shared<half4> a_tile{block_size};
        $array<half4, row_size> v[2];

        $ tid = $dispatch_x;

        a_tile[tid % block_size] = a.read(tid % block_size);
        sync_block();

        for (int i = 0; i < row_size; i++) {
            v[0][i] = b.read(i + tid * row_size);
            // v[0][i] = {i, tid, i, i};
        }
        $for(r, layer_width) {
            $half s = 0;
            for(int c = 0; c < row_size; c++) {
                $half4 t = a_tile[c + r * row_size];
                // $half4 t = {tid, r, tid, c};
                for (int i = 0; i < 4; i++) {
                    s += t[i] * v[0][c][i];
                }
            };
            v[1][r/4][r%4] = s;
        };
        for (int i = 0; i < row_size; i++) {
            c.write(i + tid * row_size, v[1][i]);
        }
    };

    timer.tic();
    auto shader = device.compile(kernel);
    print("compiled shader: {}\n", timer.toc());

    stream << a.copy_from(a_h.data()) << a1.copy_from(a1_h.data()) << b.copy_from(b_h.data()) << synchronize();

    for (int i = 0; i < 5; i++) {
        timer.tic();
        for (int i = 0; i < 100; i++) {
            stream << shader(a, b, c).dispatch(batch_size);
        }
        stream.synchronize();
        print("{}\n", timer.toc());
    }

    stream << c.copy_to(c_buffer.data()) << synchronize();

    print("[");
    for (int i = 0; i < 32; i++) {
        print("{}", c_buffer[i]);
        if (i < 31) print(", ");
    }
    print("]\n");

    float f_err = 0;
    int err_c = 0;
    for (int i = 0; i < layer_width * batch_size / 4; i++) {
        float err = abs(c_h[i] - c_buffer[i]);
        f_err = max(f_err, err);
        if (err > 0.1) {
            if (err_c < 32) {
                print("error {}: {}, {}\n", i, c_h[i], c_buffer[i]);
            }
            err_c++;
        }
    }
    print("f_err: {}\n", f_err);
    print("ok\n");

    return 0;
}