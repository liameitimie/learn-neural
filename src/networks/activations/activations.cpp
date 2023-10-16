#include <activations.h>
#include <activation_func.h>

using namespace luisa;
using namespace luisa::compute;

namespace activation {


void forward(Activation act, $half &x) {
    switch (act) {
        case ReLU: x *= (x > 0).cast<half>(); break;
        case LeakyReLU: x *= half(1) - half(0.9)*(x <= 0).cast<half>(); break;
        case Sine: x = sin(x); break;
        case None: break;
    }
}

void backward(Activation act, $half &x, $half &fwd_act) {
    switch (act) {
        case ReLU: x *= (fwd_act > 0).cast<half>(); break;
        case LeakyReLU: x *= half(1) - half(0.9)*(fwd_act <= 0).cast<half>(); break;
        case Sine: x *= cos(asin(fwd_act)).cast<half>(); break;
        case None: break;
    }
}

void forward($uint act, $half &x) {
    $switch(act) {
        $case((uint)ReLU) {
            x *= (x > 0).cast<half>();
        };
        $case((uint)LeakyReLU) {
            x *= half(1) - half(0.9)*(x <= 0).cast<half>();
        };
        $case((uint)Sine) {
            x = sin(x);
        };
        $default {};
    };
}

void apply_forward($uint act, $half4 *array, int size) {
    $switch(act) {
        $case((uint)ReLU) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < 4; j++) {
                    array[i][j] *= (array[i][j] > 0).cast<half>();
                }
            }
        };
        $case((uint)LeakyReLU) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < 4; j++) {
                    array[i][j] *= half(1) - half(0.9)*(array[i][j] <= 0).cast<half>();
                }
            }
        };
        $case((uint)Sine) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < 4; j++) {
                    array[i][j] = sin(array[i][j]);
                }
            }
        };
        $default {};
    };
}

void apply_forward($uint act, $array<luisa::half4, 16> &array) {
    $switch(act) {
        $case((uint)ReLU) {
            $for (i, 16) {
                $for (j, 4) {
                    array[i][j] *= (array[i][j] > 0).cast<half>();
                };
            };
        };
        $case((uint)LeakyReLU) {
            $for (i, 16) {
                $for (j, 4) {
                    array[i][j] *= half(1) - half(0.9)*(array[i][j] <= 0).cast<half>();
                };
            };
        };
        $case((uint)Sine) {
            $for (i, 16) {
                $for (j, 4) {
                    array[i][j] = sin(array[i][j]);
                };
            };
        };
        $default {};
    };
}

void backward($uint act, $half &x, $half &fwd_act) {
    $switch(act) {
        $case((uint)ReLU) {
            x *= (fwd_act > 0).cast<half>();
        };
        $case((uint)LeakyReLU) {
            x *= half(1) - half(0.9)*(fwd_act <= 0).cast<half>();
        };
        $case((uint)Sine) {
            x *= cos(asin(fwd_act)).cast<half>();
        };
        $default {};
    };
}

float forward_host(Activation act, float x) {
    switch (act) {
        case ReLU: return x * (x > 0);
        case LeakyReLU: return x * (x > 0 ? 1 : 0.1);
        case Sine: return sin(x);
        case None: return x;
    }
}

float backward_host(Activation act, float x, float fwd_act) {
    switch (act) {
        case ReLU: return x * (fwd_act > 0);
        case LeakyReLU: return x * (fwd_act > 0 ? 1 : 0.1);
        case Sine: return x * cos(asin(fwd_act));
        case None: return x;
    }
}

}