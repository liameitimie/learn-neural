#pragma once

#include <layer.h>
#include <adam.h>

class NGPEncodeLayer: public DiffLayer {
    Adam optim;

    int max_level_table_size;
    int levels;
    int feature_per_level = 2;

    int table_size;
    luisa::compute::Buffer<luisa::half2> _feature_table;
    luisa::compute::Buffer<luisa::half2> _feature_gradient;

    luisa::vector<int> level_offset;
    luisa::compute::Buffer<int> level_offset_buffer;
    void init_level_offset();
public:
    NGPEncodeLayer(
        int input_dim, 
        int output_dim, 
        int max_level_table_size=1<<19, 
        int levels=16, /*int feature_per_level=2*/
        AdamConfig optim_cfg = AdamConfig{}
    );
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
    virtual void reset_parameters() override;

    const auto& feature_table() { return _feature_table; }
    const auto& feature_gradient() { return _feature_gradient; }
};