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

        // print("v: [");
        // for (int i = 0; i < 32; i++) {
        //     print("{}", v[i]);
        //     if (i < 31) print(", ");
        // }
        // print("]\n");

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

    Kernel1D input_forward_kernel = []($buffer<half4> weight, $buffer<half4> input, $buffer<half4> store /*$buffer<half4> tb*/) {
        set_block_size(block_size);
        $shared<half4> a_tile{16};
        $shared<half4> b_tile{block_size};

        $ lid = $dispatch_x % 32;
        $ wid = $dispatch_x % block_size / 32;
        $ tid = $dispatch_x % block_size;
        $ bid = $dispatch_x / block_size;

        $ x_ofs = tid % 8;
        $ y_ofs = tid / 8;

        $if (tid < 16) {
            a_tile[tid] = weight.read(tid);
        };
        // $half4 tmp = input.read($dispatch_x);
        $half4 tmp = input.read(tid + bid * block_size);
        
        for (int i = 0; i < 2; i++) {
            $half t = warp_read_lane(tmp[i*2 + (~lid&1)], lid^1);
            tmp[i*2 + (~lid&1)] = t;
        }
        swap(tmp[1], tmp[2]);
        b_tile[tid/2 + tid%2*block_size/2] = tmp;
        sync_block();
        // tb.write(tid + bid*block_size,tmp);

        $half4 acc[4];
        $half4 a_frag;
        $half4 b_frag;
        $for (t, 4) {
        //     $half tt=bid;
            $for (k, input_dim) {
                a_frag = a_tile[x_ofs + k*(layer_width/4)];
                b_frag = b_tile[y_ofs + t*(tile_width/4) + k*(block_size/2)];

                // $if (tid%8 == 0) {
                //     tb.write(y_ofs + t*(tile_width/4) + k*(block_size/2) + bid*block_size, b_frag);
                // };
        //         $if (bid==1){
        //             b_frag = {tt,tt,tt,tt};
        //         };
                
        //     //     // $half4 b_frag = b_tile[y_ofs + k*(block_size/2)];
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < 4; i++) {
                        acc[j][i] += a_frag[i] * b_frag[j];
                    }
                }
            };
            for (int i = 0; i < 4; i++) {
                // $if (t == 1 & bid == 0) {
                //     acc[i] = {100, 100, 100, 100};
                // };
                store.write(
                    x_ofs + (i + y_ofs*4)*8 + (t + bid*4)*tile_size,
                    acc[i]
                );
                acc[i] = {0, 0, 0, 0};
            }
        };
        
        // $half4 t1 = input.read(tid);
        // $half4 acc[2];
        // for (int i = 0; i < 8; i++) {
        //     acc[0] = {0, 0, 0, 0};
        //     acc[1] = {0, 0, 0, 0};
        //     for (int t = 0; t <= 1; t++) {
        //         $half4 t2 = weight.read(i + t*8);
        //         for (int j = 0; j < 4; j++) {
        //             acc[0][j] += t2[j] * t1[t];
        //             acc[1][j] += t2[j] * t1[2 + t];
        //         }
        //     }
        //     acc[0].x += acc[1].x;
        //     acc[0].y += acc[1].y;
        //     acc[0].z += acc[1].z;
        //     acc[0].w += acc[1].w;
        //     store.write(i + tid*16, acc[0]);
        //     // store.write(i + 8 + tid*16, acc[1]);
        // }
    };
    Shader1D<Buffer<half4>, Buffer<half4>, Buffer<half4>/*, Buffer<half4>*/> input_forward;

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

    using mlp_network::layer_width;

    vector<half> a_h(layer_width * input_dim);
    lc::stream << mlp_network::input_weight.copy_to(a_h.data()) << synchronize();

    
    print("a_h: [");
    for (int i = 0; i < 74; i++) {
        print("{}:{}", i, a_h[i]);
        if (i < 73) print(", ");
    }
    print("]\n\n");

    const uint batch_size = 1920 * 1080;

    vector<half> b_h(input_dim * batch_size);
    vector<half> b_tmp(input_dim * batch_size);
    vector<float> c_h(layer_width * batch_size);
    vector<half> c_buffer(layer_width * batch_size);

    for (int i = 0; i < input_dim * batch_size; i++) {
        b_h[i] = pcg32::next_float();
    }
    
    print("b_h: [");
    for (int i = 0; i < 32; i++) {
        print("{}", b_h[i]);
        if (i < 31) print(", ");
    }
    print("]\n\n");

    timer.tic();

    for (int y = 0; y < batch_size; y++) {
        for (int k = 0; k < input_dim; k++) {
            for (int x = 0; x < layer_width; x++) {
                c_h[x + y * layer_width] += a_h[x + k * layer_width] * b_h[k + y * input_dim];
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

    auto b = lc::device.create_buffer<half4>(input_dim * batch_size / 4);
    auto tb = lc::device.create_buffer<half4>(input_dim * batch_size / 4);
    auto c = lc::device.create_buffer<half4>(layer_width * batch_size / 4);

    lc::stream << b.copy_from(b_h.data()) << synchronize();

    for (int i = 0; i < 5; i++) {
        timer.tic();
        for (int i = 0; i < 100; i++) {
            lc::stream << mlp_network::input_forward(mlp_network::input_weight, b, c).dispatch(batch_size / 2);
        }
        lc::stream.synchronize();
        print("{}\n", timer.toc());
    }
    

    lc::stream << c.copy_to(c_buffer.data()) << synchronize();
    // lc::stream << tb.copy_to(b_tmp.data()) << synchronize();

    // int ttt=0;
    // for (int i = 0; i < batch_size; i++) {
    //     for (int j = 0; j < input_dim; j++) {
    //         int idx1 = j + i * input_dim;
    //         int idx2 = i/512*1024 + i%512 + j*512;
    //         // int idx2 = i%4 + j*4 + i/4*8;
    //         if (b_h[idx1] != b_tmp[idx2]) {
    //             if(ttt<32)
    //                 print("error ({},{}) {}, {}: {}, {}\n", i, j, idx1, idx2, b_h[idx1], b_tmp[idx2]);
    //             ttt++;
    //         }
    //     }
    // }

    
    print("c_buffer: [");
    for (int i = 0; i < 32; i++) {
        print("{}", c_buffer[i]);
        if (i < 31) print(", ");
    }
    print("]\n\n");

    float f_err = 0;
    int err_c = 0;

    // int l = 0, r = -1;
    for (int i = 0; i < layer_width * batch_size; i++) {
        float err = abs(c_h[i] - c_buffer[i]);
        f_err = max(f_err, err);
        if (err > 0.001) {
    //         if(l <= r && r-l>10 && i<(1<<18)) print("[{}, {}]\n", l, r);
    //         l = i+1;


            if (err_c < 32) {
                print("error {}: {}, {}\n", i, c_h[i], c_buffer[i]);
            }
            err_c++;
        }
    //     else r = i;
    }
    // if(l <= r-10) print("[{}, {}]\n", l, r);
    print("f_err: {}\n", f_err);
    print("err_c: {}\n", err_c);
    print("ok\n");
    return 0;
}