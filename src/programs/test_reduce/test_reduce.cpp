#include <luisa/luisa-compute.h>
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

    Kernel1D reduce = []($buffer<float4> in, $buffer<float> out) {
        set_block_size(256);
        $shared<float4> smem{256};
        $uint tid = $dispatch_x;
        smem[tid] = in.read(tid) + in.read(tid + 256);
        sync_block();
        for (int s = 128; s > 0; s >>= 1) {
            $if (tid < s) {
                smem[tid] += smem[tid + s];
            };
            sync_block();
        }
        // $float4 t;
        // $if (tid < 32) {
        //     t = smem[tid];
        //     t = warp_active_sum(t);
        //     // for (int s = 32; s > 0; s >>= 1) {
        //     //     smem[tid] += smem[tid + s];
        //     // }
        // };
        $if (tid == 0) {
            $float4 t = smem[0];
            out.write(0, t.x + t.y + t.z + t.w);
        };
    };
    Kernel1D reduce1 = []($buffer<float4> in, $buffer<float> out) {
        set_block_size(256);
        $shared<float4> smem{8};
        $uint tid = $dispatch_x;
        $float4 t = in.read(tid) + in.read(tid + 256);
        t = warp_active_sum(t);
        $if (tid%32 == 0) {
            smem[tid/32] = t;
        };
        sync_block();
        $if (tid < 8) {
            t = smem[tid];
            t = warp_active_sum(t);
        };
        $if (tid == 0) {
            out.write(0, t.x + t.y + t.z + t.w);
        };
    };

    auto reduce_shader = global::device().compile(reduce);
    auto reduce_shader1 = global::device().compile(reduce1);

    auto in_d = global::device().create_buffer<float4>(512);
    auto out_d = global::device().create_buffer<float>(1);

    vector<float> in_h(2048);
    float sum_h = 0;
    float sum_d;

    for (float &x: in_h) {
        x = pcg32::next_float();
        sum_h += x;
    }

    global::stream() << in_d.copy_from(in_h.data()) << synchronize();

    Clock timer;

    for (int i = 0; i < 10; i++) {
        CommandList cmd_list;
        // timer.tic();
        for (int j = 0; j < 100000; j++) {
            // global::stream() << reduce_shader(in_d, out_d).dispatch(256);
            cmd_list << reduce_shader(in_d, out_d).dispatch(256);
        }
        timer.tic();
        global::stream() << cmd_list.commit();
        global::stream().synchronize();
        print("{}\n", timer.toc());
    }

    global::stream() << out_d.copy_to(&sum_d) << synchronize();

    print("sum_h: {}\nsum_d: {}\n", sum_h, sum_d);
    return 0;
}