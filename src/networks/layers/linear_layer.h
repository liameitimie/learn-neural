#pragma once

#include <layer.h>
#include <activations.h>
#include <adam.h>

float weight_scale_siren(int input_dim);
float weight_scale_xavier(int input_dim, int output_dim);

class LinearLayer: public DiffLayer {
    bool use_bias;
    activation::Activation act;
    float weight_scale;
    // float input_scale;
    Adam optim;

    luisa::compute::BufferView<luisa::half4> _weight;
    luisa::compute::BufferView<luisa::half4> _bias;

    luisa::compute::BufferView<luisa::half4> _weight_grad;
    luisa::compute::BufferView<luisa::half4> _bias_grad;

    luisa::compute::Buffer<luisa::half4> param_buffer;
    luisa::compute::Buffer<luisa::half4> grad_buffer;
public:
    LinearLayer(int input_dim, int output_dim, bool use_bias, activation::Activation act, float w_scale, AdamConfig optim_cfg = AdamConfig{});

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
    virtual int arena_size(int batch_size) override;
    int gradweight_arena_size(int batch_size);
    int gradbias_arena_size(int batch_size);

    virtual void reset_parameters() override;

    const auto& weight() { return _weight; }
    const auto& bias() { return _bias; }
    const auto& weight_grad() { return _weight_grad; }
    const auto& bias_grad() { return _bias_grad; }
    auto activation() { return act; }
};