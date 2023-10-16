#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <frequency_encode_layer.h>
#include <ngp_encode_layer.h>
#include <global.h>

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

    NGPEncodeLayer layer(3, 32);
    const uint batch_size = 1920*1080;
    auto f_in = global::device().create_buffer<half4>(layer.input_dim() * batch_size / 4);
    auto f_out = global::device().create_buffer<half4>(layer.output_dim() * batch_size / 4);

    vector<half> f_in_h(layer.input_dim() * batch_size);
    vector<float> f_out_h(layer.output_dim() * batch_size);
    vector<half> f_out_buffer(layer.output_dim() * batch_size);

    for (auto &x : f_in_h) x = pcg32::next_float() - 0.5;

    global::stream() << f_in.copy_from(f_in_h.data()) << synchronize();

    Clock timer;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 100; j++) {
            layer.forward(f_in, f_out);
        }
        timer.tic();
        global::stream() << global::cmd_list().commit();
        global::stream().synchronize();
        print("{}\n", timer.toc());
    }

    return 0;
}