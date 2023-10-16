#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <linear_layer.h>
#include <global.h>
#include <activation_func.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

namespace pcg32 {
    ulong state = 0x853c49e6748fea9bull;
    ulong inc = 0xda3e39cb94b95bdbull;
    ulong mul = 0x5851f42d4c957f2dull;

    uint next_uint() {
        uint t1 = ((state >> 18u) ^ state) >> 27u;
        uint t2 = state >> 59u;
        state = state * mul + inc;
        return (t1 >> t2) | (t1 << ((~t2 + 1u) & 31));
    }
    float next_float() {
        union {
            uint u;
            float f;
        } x;
        x.u = (next_uint() >> 9) | 0x3f800000u;
        return x.f - 1;
    }
}

template<typename T>
void print_vec(vector<T> &v, string name, int n = -1) {
    if (n == -1) n = v.size();
    n = min(n, (int)v.size());
    print("{}: [", name);
    for (int i = 0; i < n; i++) {
        print("{}, ", v[i]);
    }
    print("]\n");
}

template<typename T1, typename T2>
void compare_vec(vector<T1> &v1, vector<T2> &v2) {
    if (v1.size() != v2.size()) {
        print("compare different size vec\n");
        exit(0);
    }
    int n = v1.size();
    float f_err = 0;
    int err_c = 0;
    for (int i = 0; i < n; i++) {
        float t1 = v1[i];
        float t2 = v2[i];
        float err = abs(t1 - t2);
        f_err = max(f_err, err);
        if (err > 0.01) {
            if (err_c < 32) {
                print("error {}: {}, {}\n", i, t1, t2);
            }
            err_c++;
        }
    }
    print("f_err: {}\n", f_err);
    print("err_c: {}\n", err_c);
    print("ok\n");
}


int main(int argc, char** argv) {
    global::init(argv[0]);

    LinearLayer layer(32, 32, true, activation::Sine, 0.5);
    const uint batch_size = 16384;
    const uint pad_output_dim = (layer.output_dim() + 3) / 4 * 4;

    global::stream() << global::cmd_list().commit();

    auto f_in = global::device().create_buffer<half4>(layer.input_dim() * batch_size / 4);
    auto f_out = global::device().create_buffer<half4>(layer.output_dim() * batch_size / 4);

    vector<half> w_h(layer.input_dim() * pad_output_dim);
    vector<half> b_h(pad_output_dim);
    vector<half> f_in_h(layer.input_dim() * batch_size);
    vector<float> f_out_h(layer.output_dim() * batch_size);
    vector<half> f_out_buffer(layer.output_dim() * batch_size);

    for (auto &x : f_in_h) x = pcg32::next_float() - 0.5;

    global::stream()
        << layer.weight().copy_to(w_h.data())
        << f_in.copy_from(f_in_h.data());
    
    if (layer.bias()) {
        global::stream() << layer.bias().copy_to(b_h.data());
    }
    global::stream().synchronize();

    print_vec(w_h, "weight", 32);
    if (layer.bias()) {
        print_vec(b_h, "bias", 32);
    }

    Clock timer;

    timer.tic();
    print("calc ref fwd result: ");
    for (int k = 0; k < layer.input_dim(); k++) {
        for (int x = 0; x < layer.output_dim(); x++) {
            float tmp = w_h[x + k*pad_output_dim];
            for (int y = 0; y < batch_size; y++) {
                f_out_h[y + x*batch_size] += tmp * f_in_h[y + k*batch_size];
            }
        }
    }
    if (layer.bias()) {
        for (int x = 0; x < layer.output_dim(); x++) {
            for (int y = 0; y < batch_size; y++) {
                f_out_h[y + x*batch_size] += b_h[x];
            }
        }
    }
    if (layer.activation() != activation::None) {
        for (auto &x: f_out_h) x = activation::forward_host(layer.activation(), x);
    }
    print("{} ms\n", timer.toc());

    global::stream().synchronize();
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 100; j++) {
            layer.forward(f_in, f_out);
        }
        timer.tic();
        global::stream() << global::cmd_list().commit();
        global::stream().synchronize();
        print("{}\n", timer.toc());
    }
    
    global::stream() << f_out.copy_to(f_out_buffer.data()) << synchronize();

    print_vec(f_out_h, "f_out_h", 32);
    print_vec(f_out_buffer, "f_out_d", 32);

    compare_vec(f_out_h, f_out_buffer);


    // auto g_out = global::device().create_buffer<half4>(layer.output_dim() * batch_size / 4);
    vector<Buffer<half4>> g_outs(500);
    for (auto &x: g_outs) x = global::device().create_buffer<half4>(layer.output_dim() * batch_size / 4);

    auto g_in = global::device().create_buffer<half4>(layer.input_dim() * batch_size / 4);
    auto arena = global::device().create_buffer<half4>(layer.arena_size(batch_size) / 4);
    // auto arena = global::device().create_buffer<half4>(layer.output_dim() * batch_size / 4);

    vector<half> g_out_h(layer.output_dim() * batch_size);
    vector<float> g_out_tmp(layer.output_dim() * batch_size);
    vector<float> g_in_h(layer.input_dim() * batch_size);
    vector<half> g_in_buffer(layer.input_dim() * batch_size);
    vector<float> g_w_h(layer.input_dim() * pad_output_dim);
    vector<half> g_w_buffer(layer.input_dim() * pad_output_dim);
    vector<float> g_b_h(pad_output_dim);
    vector<half> g_b_buffer(pad_output_dim);

    for (auto &x : g_out_tmp) x = pcg32::next_float()*0.1 - 0.05;
    // for (auto &x : g_out_tmp) x = pcg32::next_float()*0.5;
    for (int i = 0; i < g_out_h.size(); i++) {
        g_out_h[i] = g_out_tmp[i];
    }

    // global::stream() << g_out.copy_from(g_out_h.data()) << synchronize();
    for (auto &x: g_outs) global::stream() << x.copy_from(g_out_h.data());
    global::stream().synchronize();

    
    print("calc ref bwd result: ");
    timer.tic();
    if (layer.activation() != activation::None) {
        int i = 0;
        for (auto &x: g_out_tmp) {
            x = activation::backward_host(layer.activation(), x, f_out_h[i]);
            i++;
        }
    }

    for (int x = 0; x < layer.input_dim(); x++) {
        for (int k = 0; k < layer.output_dim(); k++) {
            float tmp = w_h[k + x*pad_output_dim];
            for (int y = 0; y < batch_size; y++) {
                g_in_h[y + x*batch_size] += tmp * g_out_tmp[y + k*batch_size];
            }
        }
    }

    for (int y = 0; y < layer.input_dim(); y++) {
        for (int x = 0; x < layer.output_dim(); x++) {
            for (int k = 0; k < batch_size; k++) {
                g_w_h[x + y*pad_output_dim] += (float)g_out_tmp[k + x*batch_size] * (float)f_in_h[k + y*batch_size];
            }
        }
    }

    if (layer.bias()) {
        for (int x = 0; x < layer.output_dim(); x++) {
            for (int k = 0; k < batch_size; k++) {
                g_b_h[x] += g_out_tmp[k + x*batch_size];
            }
        }
    }

    print("{} ms\n", timer.toc());

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 100; j++) {
            layer.backward(f_in, f_out, g_outs[j + i*100], g_in, arena);
            // layer.backward(f_in, f_out, g_out, g_in, &arena);
        }
        timer.tic();
        global::stream() << global::cmd_list().commit();
        global::stream().synchronize();
        print("{}\n", timer.toc());
    }

    global::stream() << g_in.copy_to(g_in_buffer.data())
        << layer.weight_grad().copy_to(g_w_buffer.data());
    if (layer.bias()) {
        global::stream() << layer.bias_grad().copy_to(g_b_buffer.data());
    }
    global::stream().synchronize();

    print_vec(g_in_h, "g_in_h", 32);
    print_vec(g_in_buffer, "g_in_d", 32);

    compare_vec(g_in_h, g_in_buffer);

    print_vec(g_w_h, "g_w_h", 32);
    print_vec(g_w_buffer, "g_w_d", 32);

    compare_vec(g_w_h, g_w_buffer);

    print_vec(g_b_h, "g_b_h", 32);
    print_vec(g_b_buffer, "g_b_d", 32);

    compare_vec(g_b_h, g_b_buffer);

    return 0;
}