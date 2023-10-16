#pragma once

#include <luisa/dsl/sugar.h>

inline $float as_uniform($uint x) {
    return ((x >> 9) | 0x3f800000u).as<float>() - 1.0f;
}

inline $float2 sobol_2d($uint i) {
    auto m0 = [](luisa::uint x) { return 1u << (31 - x); };
    auto m1 = [](luisa::uint x) {
        luisa::uint m = 1u << 31;
        for(int i = 0; i < 5; i++) {
            m |= (x >> i & 1) * (m >> (1 << i));
        }
        return m;
    };
    $uint v0 = 0;
    $uint v1 = 0;
    i ^= i >> 1;
    for(int j = 0; j < 32; j++) {
        $if((i >> j & 1) != 0) {
            v0 ^= m0(j);
            v1 ^= m1(j);
        };
    }
    return make_float2(as_uniform(v0), as_uniform(v1));
};

inline $uint2 tea($uint v0, $uint v1) {
    $uint s0 = 0;
    for (int i = 0; i < 8; i++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return $uint2(v0, v1);
};
