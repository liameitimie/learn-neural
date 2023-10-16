#include <luisa/dsl/sugar.h>
#include <frequency_encode_layer.h>
#include <global.h>

using namespace luisa;
using namespace luisa::compute;

#define PI 3.14159265358979323846f

FrequencyEncodeLayer::FrequencyEncodeLayer(int input_dim, int output_dim):
    DiffLayer(input_dim, output_dim)
{}

Shader1D<uint, uint, uint, Buffer<half4>, Buffer<half4>> encode_shader;

void FrequencyEncodeLayer::forward(const BufferView<half4> input, BufferView<half4> output) {
    if (!encode_shader) {
        Kernel1D encode = []($uint batch_size, $uint in_dim, $uint out_dim, $buffer<half4> input, $buffer<half4> output) {
            set_block_size(256);
            $uint tid = $dispatch_x;
            $float4 in;
            $half4 out;
            $for (t, in_dim) {
                in = input.read(tid + t*batch_size/4);
                $for (p, out_dim/in_dim) {
                    // for (int i = 0; i < 4; i++) {
                    //     out[i] = sin(in[i] * (1 << (p/2)) * PI + (p%2)*(PI/2));
                    // }
                    out = sin(in * ((1 << (p/2)) * PI) + (p%2)*(PI/2));
                    output.write(tid + (p + out_dim/in_dim*t)*batch_size/4, out);
                };
            };
        };
        encode_shader = global::device().compile(encode);
    }
    const uint batch_size = input.size()*4 / input_dim();
    global::cmd_list() << encode_shader(batch_size, input_dim(), output_dim(), input, output).dispatch(batch_size/4);
}

void FrequencyEncodeLayer::backward(
    const BufferView<half4> fwd_input,
    const BufferView<half4> fwd_output,
    BufferView<half4> output_grad,
    BufferView<half4> input_grad,
    BufferView<half4> arena
) {
    if (input_grad) {

    }
}

void FrequencyEncodeLayer::optimize() {

}