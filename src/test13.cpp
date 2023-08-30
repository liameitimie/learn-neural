#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

const uint input_dim = 2;
const uint output_dim = 3;
const uint output_dim_pad = 4;

namespace pcg32 {
    uint64 state = 0x853c49e6748fea9bull;
    uint64 inc = 0xda3e39cb94b95bdbull;
    uint64 mul = 0x5851f42d4c957f2dull;

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
// float rand_float() {
//     return rand() / 65536.0;
// }

namespace lc {
    Context* ctx;
    Device device;
    Stream stream;

    void init(const char *program_path) {
        ctx = new Context(program_path);
        device = ctx->create_device("dx");
        stream = device.create_stream();
    }
}

namespace mlp_network {

    const uint layer_width = 32;
    const uint hidden_layers = 5;

    Buffer<half4> weight_buffer;
    BufferView<half4> layer_weight[hidden_layers - 1]; // 32 * 32, row:l+1, col:l, col-major
    BufferView<half4> input_weight; // row:32, col:2, col-major
    BufferView<half4> output_weight; // row:4, col:32, col-major

    enum weight_type { hidden, input, output };
    int fan_in(weight_type t) {
        switch (t) {
            case hidden: return layer_width;
            case input: return input_dim;
            case output: return layer_width;
        }
    }
    int fan_out(weight_type t) {
        switch (t) {
            case hidden: return layer_width;
            case input: return layer_width;
            case output: return output_dim;
        }
    }

    void init_weight() {
        vector<half> v;
        v.reserve(layer_width * layer_width * (hidden_layers - 1) + input_dim * layer_width + output_dim_pad * layer_width);
        for (int i = 0; i < hidden_layers - 1; i++) {
            for (int j = 0; j < layer_width * layer_width; j++) {
                float s = sqrt(6.0 / (fan_in(hidden) + fan_out(hidden)));
                v.push_back(half(s * (pcg32::next_float()*2 - 1)));
            }
        }
        for (int i = 0; i < input_dim * layer_width; i++) {
            float s = sqrt(6.0 / (fan_in(input) + fan_out(input)));
            v.push_back(half(s * (pcg32::next_float()*2 - 1)));
        }
        for (int i = 0; i < output_dim_pad * layer_width; i++) {
            float s = sqrt(6.0 / (fan_in(output) + fan_out(output)));
            v.push_back(half(s * (pcg32::next_float()*2 - 1)));
        }

        lc::stream << weight_buffer.copy_from(v.data()) << synchronize();
    }

    const uint tile_width = 128;
    const uint block_size = 256;
    const uint tile_size = tile_width * layer_width / 4;

    void swap($half &a, $half &b) {
        $half t = a;
        a = b;
        b = t;
    }

    void load_a_mat($buffer<half4> &a, $shared<half4> &a_tile) {
        $ tid = $dispatch_x % block_size;
        a_tile[tid] = a.read(tid);
    }
    void load_a_mat_transpose($buffer<half4> &a, $shared<half4> &a_tile, $half4 &tmp) {
        $ lid = $dispatch_x % 32;
        $ wid = $dispatch_x % block_size / 32;
        $ tid = $dispatch_x % block_size;
        tmp = a.read(tid);
        $ idx = lid / 8;
        for (int ofs = 1; ofs < 4; ofs++) {
            $half t = warp_read_lane(tmp[idx^ofs], lid^(ofs*8));
            tmp[idx^ofs] = t;
        }
        a_tile[wid + (lid/8 + lid%8*4)*8] = tmp;
    }
    void load_b_mat_transpose($buffer<half4> &b, $shared<half4> &b_tile, $half4 *tmp) {
        $ tid = $dispatch_x % block_size;
        $ bid = $dispatch_x / block_size;

        for (int i = 0; i < 4; i++) {
            tmp[i] = b.read(tid%8 + (i + tid/8*4)*8 + bid*tile_size);
        }
        for (int i = 1; i < 4; i++) {
            for (int j = 0; j < i; j++) {
                swap(tmp[i][j], tmp[j][i]);
            }
        }
        for (int i = 0; i < 4; i++) {
            b_tile[tid/8 + (i + tid%8*4)*32] = tmp[i];
            tmp[i] = {0, 0, 0, 0};
        }
    }
    void mma($shared<half4> &a_tile, $shared<half4> &b_tile, $half4 *tmp) {
        $ tid = $dispatch_x % block_size;
        $ x_ofs = tid % 8;
        $ y_ofs = tid / 8;
        $half4 a_frag;
        $half4 b_frag;
        $for(k, layer_width) {
            a_frag = a_tile[x_ofs + k*(layer_width/4)];
            b_frag = b_tile[y_ofs + k*(tile_width/4)];
            for (int j = 0; j < 4; j++) {
                for (int i = 0; i < 4; i++) {
                    tmp[j][i] += a_frag[i] * b_frag[j];
                }
            }
        };
    }
    void store_res_mat($buffer<half4> &store, $half4 *acc) {
        $ tid = $dispatch_x % block_size;
        $ bid = $dispatch_x / block_size;
        for (int i = 0; i < 4; i++) {
            store.write(
                tid%8 + (i + tid/8*4)*8 + bid*tile_size,
                acc[i]
            );
        }
    }
    void kernel_impl($buffer<half4> &a, $buffer<half4> &b, $buffer<half4> &store, bool is_backward) {
        set_block_size(block_size);
        $shared<half4> a_tile{block_size};
        $shared<half4> b_tile{block_size * 4};
        $half4 tmp[4];

        if(!is_backward) load_a_mat(a, a_tile);
        else load_a_mat_transpose(a, a_tile, tmp[0]);
        load_b_mat_transpose(b, b_tile, tmp);
        sync_block();

        mma(a_tile, b_tile, tmp);
        store_res_mat(store, tmp);
    }
    Kernel1D layer_forward_kernel = []($buffer<half4> weight, $buffer<half4> act, $buffer<half4> store) {
        kernel_impl(weight, act, store, false);
    };
    Kernel1D layer_backward_kernel = []($buffer<half4> weight, $buffer<half4> grad, $buffer<half4> store) {
        kernel_impl(weight, grad, store, true);
    };

    Shader1D<Buffer<half4>, Buffer<half4>, Buffer<half4>> layer_forward;
    Shader1D<Buffer<half4>, Buffer<half4>, Buffer<half4>> layer_backward;

    Kernel1D input_forward_kernel = []($buffer<half4> weight, $buffer<half4> input, $buffer<half4> store) {
        set_block_size(block_size);
        $shared<half4> a_tile{16};
        $shared<half4> b_tile{block_size};

        $ lid = $dispatch_x % 32;
        $ tid = $dispatch_x % block_size;
        $ bid = $dispatch_x / block_size;

        $ x_ofs = tid % 8;
        $ y_ofs = tid / 8;

        $if (tid < 16) {
            a_tile[tid] = weight.read(tid);
        };
        $half4 tmp = input.read(tid + bid * block_size);
        
        for (int i = 0; i < 2; i++) {
            $half t = warp_read_lane(tmp[i*2 + (~lid&1)], lid^1);
            tmp[i*2 + (~lid&1)] = t;
        }
        swap(tmp[1], tmp[2]);
        b_tile[tid/2 + tid%2*block_size/2] = tmp;
        sync_block();

        $half4 acc[4];
        $half4 a_frag;
        $half4 b_frag;
        $for (t, 4) {
            $for (k, input_dim) {
                a_frag = a_tile[x_ofs + k*(layer_width/4)];
                b_frag = b_tile[y_ofs + t*(tile_width/4) + k*(block_size/2)];
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < 4; i++) {
                        acc[j][i] += a_frag[i] * b_frag[j];
                    }
                }
            };
            for (int i = 0; i < 4; i++) {
                store.write(
                    x_ofs + (i + y_ofs*4)*8 + (t + bid*4)*tile_size,
                    acc[i]
                );
                acc[i] = {0, 0, 0, 0};
            }
        };
    };

    Shader1D<Buffer<half4>, Buffer<half4>, Buffer<half4>> input_forward;

    Kernel1D output_forward_kernel = []($buffer<half4> weight, $buffer<half4> act, $buffer<half4> store) {
        set_block_size(block_size);
        $shared<half4> a_tile{32};
        $shared<half4> b_tile{block_size * 4};
        $half4 tmp[4];
        $half4 a_frag;
        $half4 b_frag;

        $ lid = $dispatch_x % 32;
        $ tid = $dispatch_x % block_size;
        $ bid = $dispatch_x / block_size;

        $if (tid < 32) {
            a_tile[tid] = weight.read(tid);
        };

        $for (k1, layer_width / 4) {
            sync_block();
            $for (i, 4) {
                b_frag = act.read(k1 + tid*8 + i*8*block_size + bid*8*block_size*4);
                $ idx = lid % 4;
                for (int ofs = 1; ofs < 4; ofs++) {
                    $half t = warp_read_lane(b_frag[idx^ofs], lid^ofs);
                    b_frag[idx^ofs] = t;
                }
                b_tile[tid/4 + i*block_size/4 + tid%4*block_size] = b_frag;
            };
            sync_block();
            $for (k, k1*4, k1*4+4) {
                a_frag = a_tile[k];
                b_frag = b_tile[tid + k%4*block_size];
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < 3; i++) {
                        tmp[j][i] += a_frag[i] * b_frag[j];
                    }
                }
            };
        };
        for (int i = 0; i < 4; i++) {
            store.write(i + tid*4 + bid*block_size*4, tmp[i]);
        }
    };

    Shader1D<Buffer<half4>, Buffer<half4>, Buffer<half4>> output_forward;

    void init() {
        const uint layer_weight_size = layer_width * layer_width / 4;
        const uint input_weight_size = input_dim * layer_width / 4;
        const uint output_weight_size = output_dim_pad * layer_width / 4;

        weight_buffer = lc::device.create_buffer<half4>(layer_weight_size * (hidden_layers - 1) + input_weight_size + output_weight_size);
        for (int i = 0; i < hidden_layers - 1; i++) {
            layer_weight[i] = weight_buffer.view(layer_weight_size * i, layer_weight_size);
        }
        input_weight = weight_buffer.view(layer_weight_size * (hidden_layers - 1), input_weight_size);
        output_weight = weight_buffer.view(layer_weight_size * (hidden_layers - 1) + input_weight_size, output_weight_size);

        init_weight();

        layer_forward = lc::device.compile(layer_forward_kernel);
        layer_backward = lc::device.compile(layer_backward_kernel);
        input_forward = lc::device.compile(input_forward_kernel);
        output_forward = lc::device.compile(output_forward_kernel);
    }
    void inference() {

    }
    void train() {

    }
}

namespace trainer {
    void init() {

    }
}

int main(int argc, char *argv[]) {
    lc::init(argv[0]);
    mlp_network::init();

    Clock timer;

    const uint layer_width = 32;

    vector<half> a_h(layer_width * output_dim_pad);
    lc::stream << mlp_network::output_weight.copy_to(a_h.data()) << synchronize();

    print("a_h: [");
    for (int i = 0; i < 32; i++) {
        print("{}:{}", i, a_h[i]);
        if (i < 31) print(", ");
    }
    print("]\n\n");

    const uint batch_size = 1920 * 1080;

    vector<half> b_h(layer_width * batch_size);
    vector<float> c_h(output_dim_pad * batch_size);
    vector<half> c_buffer(output_dim_pad * batch_size);

    for (int i = 0; i < layer_width * batch_size; i++) {
        b_h[i] = pcg32::next_float();
    }

    timer.tic();

    for (int y = 0; y < batch_size; y++) {
        for (int k = 0; k < layer_width; k++) {
            for (int x = 0; x < output_dim; x++) {
                c_h[x + y * output_dim_pad] += a_h[x + k * output_dim_pad] * b_h[k + y * layer_width];
            }
        }
    }
    print("{}\n", timer.toc());
    print("c_h: [");
    for (int i = 0; i < 32; i++) {
        print("{}", c_h[i]);
        if (i < 31) print(", ");
    }
    print("]\n\n");

    auto b = lc::device.create_buffer<half4>(layer_width * batch_size / 4);
    auto c = lc::device.create_buffer<half4>(output_dim_pad * batch_size / 4);

    lc::stream << b.copy_from(b_h.data()) << synchronize();

    for (int i = 0; i < 5; i++) {
        timer.tic();
        for (int i = 0; i < 100; i++) {
            lc::stream << mlp_network::output_forward(mlp_network::output_weight, b, c).dispatch(batch_size / 4);
        }
        lc::stream.synchronize();
        print("{}\n", timer.toc());
    }

    lc::stream << c.copy_to(c_buffer.data()) << synchronize();

    print("c_buffer: [");
    for (int i = 0; i < 32; i++) {
        print("{}", c_buffer[i]);
        if (i < 31) print(", ");
    }
    print("]\n\n");

    float f_err = 0;
    int err_c = 0;
    for (int i = 0; i < output_dim_pad * batch_size; i++) {
        float err = abs(c_h[i] - c_buffer[i]);
        f_err = max(f_err, err);
        if (err > 0.01) {
            if (err_c < 32) {
                print("error {}: {}, {}\n", i, c_h[i], c_buffer[i]);
            }
            err_c++;
        }
    }
    print("f_err: {}\n", f_err);
    print("err_c: {}\n", err_c);
    print("ok\n");
    return 0;
}