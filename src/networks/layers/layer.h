#pragma once

#include <luisa/runtime/buffer.h>

class DiffLayer {
    int _input_dim;
    int _output_dim;
public:
    DiffLayer(int input_dim, int output_dim):
        _input_dim(input_dim),
        _output_dim(output_dim) {}
    
    int input_dim() { return _input_dim; }
    int output_dim() {return _output_dim; }

    virtual void reset_parameters() {};

    virtual void forward(
        const luisa::compute::BufferView<luisa::half4> input,
        luisa::compute::BufferView<luisa::half4> output
    ) = 0;
    virtual void backward(
        const luisa::compute::BufferView<luisa::half4> fwd_input,
        const luisa::compute::BufferView<luisa::half4> fwd_output,
        luisa::compute::BufferView<luisa::half4> output_grad,
        luisa::compute::BufferView<luisa::half4> input_grad,
        luisa::compute::BufferView<luisa::half4> arena
    ) = 0;
    virtual void optimize() = 0;

    virtual int arena_size(int batch_size) { return 0; }
};