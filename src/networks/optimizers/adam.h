#pragma once

#include <luisa/runtime/buffer.h>

struct AdamConfig {
    float learning_rate = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float l2_reg = 0;
};

class Adam {
    float learning_rate = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float l2_reg = 0;
    int t = 0;

    luisa::compute::Buffer<luisa::float4> param_fp;
    luisa::compute::Buffer<luisa::float4> mt;
    luisa::compute::Buffer<luisa::float4> vt;

public:
    Adam() {};
    Adam(float learning_rate, float beta1, float beta2, float l2_reg):
        learning_rate(learning_rate),
        beta1(beta1),
        beta2(beta2),
        l2_reg(l2_reg)
    {}
    Adam(AdamConfig cfg):
        learning_rate(cfg.learning_rate),
        beta1(cfg.beta1),
        beta2(cfg.beta2),
        l2_reg(cfg.l2_reg)
    {}

    void init(luisa::compute::BufferView<luisa::half4> param);
    void optimize(
        luisa::compute::BufferView<luisa::half4> param,
        luisa::compute::BufferView<luisa::half4> grad
    );
};