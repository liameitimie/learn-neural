#pragma once

#include <luisa/runtime/buffer.h>

class Loss {
public:
    virtual void evaluate(
        int batch_size,
        luisa::compute::BufferView<luisa::half4> predict,
        luisa::compute::BufferView<luisa::half4> target,
        luisa::compute::BufferView<luisa::half4> loss,
        luisa::compute::BufferView<luisa::half4> loss_grad
    ) = 0;
};