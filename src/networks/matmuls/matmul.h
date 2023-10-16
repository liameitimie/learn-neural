#include <luisa/core/basic_types.h>
// #include <dll_export.h>

namespace luisa::compute {
    template<typename T>
    class BufferView;
}

namespace activation {
    enum Activation;
}

namespace matmuls {
    // void init();
    void mm_crr_32_x_32(
        luisa::uint x, 
        luisa::compute::BufferView<luisa::half4> a, 
        luisa::compute::BufferView<luisa::half4> b, 
        luisa::compute::BufferView<luisa::half4> d
    );
    void mm_rrr_32_x_32(
        luisa::uint x, 
        luisa::compute::BufferView<luisa::half4> a, 
        luisa::compute::BufferView<luisa::half4> b, 
        luisa::compute::BufferView<luisa::half4> d
    );
    void mma_crr_32_x_32(
        luisa::uint x, 
        luisa::compute::BufferView<luisa::half4> a, 
        luisa::compute::BufferView<luisa::half4> b,
        luisa::compute::BufferView<luisa::half4> c,
        luisa::compute::BufferView<luisa::half4> d
    );
    void mma_rrr_32_x_32(
        luisa::uint x, 
        luisa::compute::BufferView<luisa::half4> a, 
        luisa::compute::BufferView<luisa::half4> b,
        luisa::compute::BufferView<luisa::half4> c,
        luisa::compute::BufferView<luisa::half4> d
    );

    // matmul and activation result
    void act_mm_crr_32_x_32(
        activation::Activation act, 
        luisa::uint x, 
        luisa::compute::BufferView<luisa::half4> a, 
        luisa::compute::BufferView<luisa::half4> b, 
        luisa::compute::BufferView<luisa::half4> d
    );
    void act_mm_rrr_32_x_32(
        activation::Activation act, 
        luisa::uint x, 
        luisa::compute::BufferView<luisa::half4> a, 
        luisa::compute::BufferView<luisa::half4> b, 
        luisa::compute::BufferView<luisa::half4> d
    );
    void act_mma_crr_32_x_32(
        activation::Activation act, 
        luisa::uint x, 
        luisa::compute::BufferView<luisa::half4> a, 
        luisa::compute::BufferView<luisa::half4> b,
        luisa::compute::BufferView<luisa::half4> c,
        luisa::compute::BufferView<luisa::half4> d
    );
    void act_mma_rrr_32_x_32(
        activation::Activation act, 
        luisa::uint x, 
        luisa::compute::BufferView<luisa::half4> a, 
        luisa::compute::BufferView<luisa::half4> b,
        luisa::compute::BufferView<luisa::half4> c,
        luisa::compute::BufferView<luisa::half4> d
    );
}