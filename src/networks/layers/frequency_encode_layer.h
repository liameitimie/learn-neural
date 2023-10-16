#pragma once

#include <layer.h>

class FrequencyEncodeLayer: public DiffLayer {
public:
    FrequencyEncodeLayer(int input_dim, int output_dim);
    virtual void forward(
        const luisa::compute::BufferView<luisa::half4> input,
        luisa::compute::BufferView<luisa::half4> output
    ) override;
    virtual void backward(
        const luisa::compute::BufferView<luisa::half4> fwd_input,
        const luisa::compute::BufferView<luisa::half4> fwd_output,
        luisa::compute::BufferView<luisa::half4> output_grad,
        luisa::compute::BufferView<luisa::half4> input_grad,
        luisa::compute::BufferView<luisa::half4> arena
    ) override;
    virtual void optimize() override;
};