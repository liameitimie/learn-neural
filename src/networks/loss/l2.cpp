#include <luisa/dsl/sugar.h>
#include <l2.h>
#include <global.h>

using namespace luisa;
using namespace luisa::compute;

Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>, Buffer<half4>> evaluate_shader;
Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>> evaluate_shader1;

void evaluate_impl($uint &batch_size, $buffer<half4> &predict, $buffer<half4> &target, $buffer<half4> *loss, $buffer<half4> &loss_grad) {
    set_block_size(256);
    $uint tid = $dispatch_x;

    $float4 p = predict.read(tid);
    $float4 t = target.read(tid);
    $half4 g;
    $half4 l;

    for (int i = 0; i < 4; i++) {
        $float s = p[i] - t[i];
        // g[i] = 2 * s;
        // l[i] = s * s;
        g[i] = 2 * s / batch_size;
        l[i] = s * s / batch_size;
    }
    loss_grad.write(tid, g);
    if (loss) {
        (*loss).write(tid, l);
    }
}

void L2Loss::evaluate(int batch_size, BufferView<half4> predict, BufferView<half4> target, BufferView<half4> loss, BufferView<half4> loss_grad){
    if (loss) {
        if (!evaluate_shader) {
            Kernel1D evaluate = []($uint batch_size, $buffer<half4> predict, $buffer<half4> target, $buffer<half4> loss, $buffer<half4> loss_grad) {
                evaluate_impl(batch_size, predict, target, &loss, loss_grad);
            };
            evaluate_shader = global::device().compile(evaluate);
        }
        global::cmd_list() << evaluate_shader(batch_size, predict, target, loss, loss_grad).dispatch(predict.size());
    }
    else {
        if (!evaluate_shader1) {
            Kernel1D evaluate = []($uint batch_size, $buffer<half4> predict, $buffer<half4> target, $buffer<half4> loss_grad) {
                evaluate_impl(batch_size, predict, target, nullptr, loss_grad);
            };
            evaluate_shader1 = global::device().compile(evaluate);
        }
        global::cmd_list() << evaluate_shader1(batch_size, predict, target, loss_grad).dispatch(predict.size());
    }
}