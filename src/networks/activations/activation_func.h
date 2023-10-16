#pragma once

#include <luisa/dsl/sugar.h>
#include <activations.h>

namespace activation {

void forward(Activation act, $half &x);
void backward(Activation act, $half &x, $half &fwd_act);

void forward($uint act, $half &x);
void apply_forward($uint act, $half4 *array, int size);
void apply_forward($uint act, $array<luisa::half4, 16> &array);
void backward($uint act, $half &x, $half &fwd_act);

float forward_host(Activation act, float x);
float backward_host(Activation act, float x, float fwd_act);

}