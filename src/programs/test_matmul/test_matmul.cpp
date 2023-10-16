#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <global.h>
#include <matmul.h>
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

void act_backward_impl(activation::Activation act, $buffer<half4> &fwd_out, $buffer<half4> &g_out, $buffer<half4> &g_in) {
    set_block_size(256);
    $uint tid = $dispatch_x;
    $half4 g = g_out.read(tid);
    $half4 o = fwd_out.read(tid);
    for (int i = 0; i < 4; i++) {
        activation::backward(act, g[i], o[i]);
    }
    g_in.write(tid, g);
};
void act_forward_impl(activation::Activation act, $buffer<half4> &input, $buffer<half4> &output) {
    set_block_size(256);
    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;
    for (int i = 0; i < 32; i++) {
        $half4 t = input.read(tid + i*256 + bid*256*32);
        for (int i = 0; i < 4; i++) {
            activation::forward(act, t[i]);
        }
        output.write(tid + i*256 + bid*256*32, t);
    }
};

Shader1D<Buffer<half4>, Buffer<half4>, Buffer<half4>> act_backward_shader;
Shader1D<Buffer<half4>, Buffer<half4>> act_forward_shader;

#define test_trans_a 0
#define test_mma 0
#define test_out_act 1
#define test_act (1 && !test_out_act)

int main(int argc, char** argv) {
    global::init(argv[0]);
    // matmuls::init();

    const activation::Activation act = activation::Sine;
    const uint layer_width = 32;
    const uint batch_size = 1920*1080;
    const uint a_size = layer_width * layer_width;
    const uint b_size = layer_width * batch_size;
    const uint c_size = layer_width * batch_size;
    const uint d_size = layer_width * batch_size;

    {
        Kernel1D kernel1 = [&]($buffer<half4> fwd_out, $buffer<half4> g_out, $buffer<half4> g_in) {
            act_backward_impl(act, fwd_out, g_out, g_in);
        };
        Kernel1D kernel2 = [&]($buffer<half4> in, $buffer<half4> out) {
            act_forward_impl(act, in, out);
        };
        act_backward_shader = global::device().compile(kernel1);
        act_forward_shader = global::device().compile(kernel2);
    }

    auto a = global::device().create_buffer<half4>(a_size / 4);
    auto b = global::device().create_buffer<half4>(b_size / 4);
    auto c = global::device().create_buffer<half4>(c_size / 4);
    auto d = global::device().create_buffer<half4>(d_size / 4);

    vector<half> a_h(a_size);
    vector<half> b_h(b_size);
    vector<half> c_h(c_size);
    vector<float> d_h(d_size);
    vector<half> d_buffer(d_size);

    for (auto &x : a_h) x = pcg32::next_float() - 0.5;
    for (auto &x : b_h) x = pcg32::next_float() - 0.5;
    for (auto &x : c_h) x = pcg32::next_float() - 0.5;

    for (int k = 0; k < layer_width; k++) {
        for (int x = 0; x < layer_width; x++) {
#if test_trans_a
            float tmp = a_h[k + x*layer_width];
#else
            float tmp = a_h[x + k*layer_width];
#endif
            for (int y = 0; y < batch_size; y++) {
                d_h[y + x*batch_size] += tmp * b_h[y + k*batch_size];
            }
        }
    }
#if test_mma
    for (int i = 0; i < d_size; i++) d_h[i] += c_h[i];
#endif
#if test_act || test_out_act
    for (auto &x: d_h) x = activation::forward_host(act, x);
#endif

    global::stream() 
        << a.copy_from(a_h.data()) 
        << b.copy_from(b_h.data())
        << c.copy_from(c_h.data())
        << synchronize();

    auto call_matmul = [&]() {
#if test_act
    #if test_mma
        #if test_trans_a
            matmuls::act_mma_rrr_32_x_32(act, batch_size, a.view(), b.view(), c.view(), d.view());
        #else
            matmuls::act_mma_crr_32_x_32(act, batch_size, a.view(), b.view(), c.view(), d.view());
        #endif
    #else
        #if test_trans_a
            matmuls::act_mm_rrr_32_x_32(act, batch_size, a.view(), b.view(), d.view());
        #else
            matmuls::act_mm_crr_32_x_32(act, batch_size, a.view(), b.view(), d.view());
        #endif
    #endif
#else
    #if test_mma
        #if test_trans_a
            matmuls::mma_rrr_32_x_32(batch_size, a.view(), b.view(), c.view(), d.view());
        #else
            matmuls::mma_crr_32_x_32(batch_size, a.view(), b.view(), c.view(), d.view());
        #endif
    #else
        #if test_trans_a
            matmuls::mm_rrr_32_x_32(batch_size, a.view(), b.view(), d.view());
        #else
            matmuls::mm_crr_32_x_32(batch_size, a.view(), b.view(), d.view());
        #endif
    #endif
#endif

#if test_out_act
            global::stream() << act_forward_shader(d.view(), d.view()).dispatch(batch_size / 4);
#endif
    };

    call_matmul();
    global::stream().synchronize();

    Clock timer;
    for (int i = 0; i < 10; i++) {
        timer.tic();
        for (int j = 0; j < 100; j++) {
            call_matmul();
        }
        global::stream().synchronize();
        print("{}\n", timer.toc());
    }

    global::stream() << d.copy_to(d_buffer.data()) << synchronize();

    print("d_h: [");
    for (int i = 0; i < 32; i++) {
        print("{}", d_h[i]);
        if (i < 31) print(", ");
    }
    print("]\n");

    print("d_buffer: [");
    for (int i = 0; i < 32; i++) {
        print("{}", d_buffer[i]);
        if (i < 31) print(", ");
    }
    print("]\n");

    float f_err = 0;
    float f_err2 = 0;
    int err_c = 0;
    int i = 0;
    for (float x: d_buffer) {
        float y = d_h[i];
        float err = abs(y - x) / max(abs(x), abs(y));
        float err2 = abs(y - x);

        if (err > f_err) {
            print("!inc error {}: {}, {}; f_err: {}\n", i, y, x, err);
        }

        f_err = max(f_err, err);
        f_err2 = max(f_err2, err2);
        if (err > 0.005 || err2 > 0.05) {
            if (err_c < 32) {
                print("error {}: {}, {}\n", i, y, x);
            }
            err_c++;
        }
        i++;
    }
    print("f_err: {}\n", f_err);
    print("f_err2: {}\n", f_err2);
    print("err_c: {}\n", err_c);
    print("ok\n");
    
    return 0;
}
