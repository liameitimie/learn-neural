#include <luisa/dsl/sugar.h>
#include <adam.h>
#include <global.h>

using namespace luisa;
using namespace luisa::compute;

Shader1D<Buffer<half4>, Buffer<float4>, Buffer<float4>, Buffer<float4>> init_buffer_shader;
Shader1D<Buffer<half4>, Buffer<half4>, uint, Buffer<float4>, Buffer<float4>, Buffer<float4>, float, float, float, float> adam_optimize_shader;

void Adam::init(BufferView<half4> param) {
    if (!init_buffer_shader) {
        Kernel1D init_buffer = []($buffer<half4> param, $buffer<float4> param_fp, $buffer<float4> mt, $buffer<float4> vt) {
            $uint tid = $dispatch_x;
            $float4 w = param.read(tid);
            param_fp.write(tid, w);
            mt.write(tid, make_float4(0));
            vt.write(tid, make_float4(0));
        };
        init_buffer_shader = global::device().compile(init_buffer);
    }
    int n_param = param.size();
    param_fp = global::device().create_buffer<float4>(n_param);
    mt = global::device().create_buffer<float4>(n_param);
    vt = global::device().create_buffer<float4>(n_param);

    global::cmd_list() << init_buffer_shader(param, param_fp, mt, vt).dispatch(n_param);
}

void Adam::optimize(BufferView<half4> param, BufferView<half4> grad) {
    if (!adam_optimize_shader) {
        Kernel1D optimize_kernel = [](
            $buffer<half4> param, 
            $buffer<half4> grad, 
            $uint t, 
            $buffer<float4> param_fp, 
            $buffer<float4> mt, 
            $buffer<float4> vt,
            $float learning_rate,
            $float beta1,
            $float beta2,
            $float l2_reg
        ) {
            $uint tid = $dispatch_x;
            $float4 w = param_fp.read(tid);
            $half4 g = grad.read(tid);
            $float4 m = mt.read(tid);
            $float4 v = vt.read(tid);
            $float tg;
            for (int i = 0; i < 4; i++) {
                tg = g[i];
                tg += l2_reg * w[i];
                m[i] = beta1*m[i] + (1 - beta1)*tg;
                v[i] = beta2*v[i] + (1 - beta2)*tg*tg;
            }
            mt.write(tid, m);
            vt.write(tid, v);
            
            for (int i = 0; i < 4; i++) {
                m[i] = m[i] / (1 - pow(beta1, t.cast<float>()));
                v[i] = v[i] / (1 - pow(beta2, t.cast<float>()));

                w[i] = w[i] - learning_rate*m[i] / (sqrt(v[i]) + 1e-8f);
            }
            param_fp.write(tid, w);
            $half4 tw = w;
            param.write(tid, tw);
        };
        adam_optimize_shader = global::device().compile(optimize_kernel);
    }
    t++;
    global::cmd_list() << adam_optimize_shader(param, grad, t, param_fp, mt, vt, learning_rate, beta1, beta2, l2_reg).dispatch(param.size());
}