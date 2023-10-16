#include <luisa/dsl/sugar.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <global.h>
#include <gpu_rands.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

namespace pcg32 {
    ulong state = 0x853c49e6748fea9bull;
    ulong inc = 0xda3e39cb94b95bdbull;
    ulong mul = 0x5851f42d4c957f2dull;

    uint next_uint() {
        uint t1 = ((state >> 18u) ^ state) >> 27u;
        uint t2 = state >> 59u;
        state = state * mul + inc;
        return (t1 >> t2) | (t1 << ((~t2 + 1u) & 31));
    }
    float next_float() {
        union {
            uint u;
            float f;
        } x;
        x.u = (next_uint() >> 9) | 0x3f800000u;
        return x.f - 1;
    }
}

int main(int argc, char** argv) {
    global::init(argv[0]);

    Kernel1D kernel = []($buffer<int> steps, $buffer<int> out) {
        set_block_size(256);
        $shared<int> cnt{1};

        $int tid = $dispatch_x % 256;
        $int step = steps.read(tid);
        $int idx;

        $for (t, step) {
            for (int i = 0; i < 8; i++) {
                $if (tid/32 == i) {
                    idx = cnt.atomic(0).fetch_add(1);
                    out.write(idx, tid);
                };
                sync_block();
            }
        };
    };
    auto shader = global::device().compile(kernel);
    auto steps = global::device().create_buffer<int>(256);
    auto out = global::device().create_buffer<int>(256 * 5);

    vector<int> steps_h(256);
    vector<int> out_h(256 * 5, -1);
    for (int &x: steps_h) x = pcg32::next_uint() % 4;

    for (int i = 0; i < steps_h.size(); i++) {
        print("{}:{}, ", i, steps_h[i]);
    }
    print("\n\n");
    
    global::stream()
        << steps.copy_from(steps_h.data())
        << out.copy_from(out_h.data())
        << shader(steps, out).dispatch(256)
        << out.copy_to(out_h.data())
        << synchronize();

    int lst = -1;
    for (int x: out_h) {
        if (x == -1) break;
        if (x < lst) print("\n\n");
        print("{}, ", x);
        lst = x;
    }
    print("\n");
    return 0;
}
