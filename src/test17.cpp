#include <luisa/luisa-compute.h>
#include <luisa/gui/window.h>
#include <stb/stb_image.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

#define use_input_encode 1
#define use_instant_ngp (use_input_encode && 1)
#define use_ema_weight 1
#define pi 3.1415926535897f

namespace pcg32 {
    uint next_uint();
    float next_float();
}

$float as_uniform($uint x) {
    return ((x >> 9) | 0x3f800000u).as<float>() - 1.0f;
}

$float2 sobol_2d($uint i) {
    auto m0 = [](uint x) { return 1u << (31 - x); };
    auto m1 = [](uint x) {
        uint m = 1u << 31;
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

$uint tea($uint v0, $uint v1) {
    $uint s0 = 0;
    for (int i = 0; i < 8; i++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
};

namespace lc {
    Context* ctx;
    Device device;
    Stream stream;

    void init(const char *program_path) {
        ctx = new Context(program_path);
        device = ctx->create_device("dx");
        stream = device.create_stream(StreamTag::GRAPHICS);
    }
}

const uint input_dim = 2;
const uint output_dim = 3;
const uint output_dim_pad = 4;

enum class Activation {
	ReLU,
	LeakyReLU,
	Sine,
	None,
};
const Activation activation = Activation::ReLU;

void activation_forward($half &x) {
    switch (activation) {
        case Activation::ReLU: x *= (x > 0).cast<half>();break;
        case Activation::LeakyReLU: x *= half(1) - half(0.9)*(x <= 0).cast<half>();break;
        case Activation::Sine: x = sin(x);break;
        case Activation::None:break;
    }
}
void apply_activation_forward($half4 &acc) {
    for (int i = 0; i < 4; i++) {
        activation_forward(acc[i]);
    }
}
void activation_backward($half &x, $half &fwd_act) {
    switch (activation) {
        case Activation::ReLU: x *= (fwd_act > 0).cast<half>();break;
        case Activation::LeakyReLU: x *= half(1) - half(0.9)*(fwd_act <= 0).cast<half>();break;
        case Activation::Sine: x *= cos(asin(fwd_act)).cast<half>();break;
        case Activation::None:break;
    }
}
void apply_activation_backward($half4 &acc, $half4 &fwd_act) {
    for (int i = 0; i < 4; i++) {
        activation_backward(acc[i], fwd_act[i]);
    }
}

void activation_forward_host(float &x) {
    switch (activation) {
        case Activation::ReLU: x *= (x > 0);break;
        case Activation::LeakyReLU: x *= (x > 0 ? 1 : 0.1);break;
        case Activation::Sine: x = sin(x);break;
        case Activation::None:break;
    }
}
void activation_backward_host(float &x, float fwd_act) {
    switch (activation) {
        case Activation::ReLU: x *= (fwd_act > 0);break;
        case Activation::LeakyReLU: x *= (fwd_act > 0 ? 1 : 0.1);break;
        case Activation::Sine: x *= sin(asin(fwd_act));break;
        case Activation::None:break;
    }
}

enum class Loss {
    L2,
    RelativeL2
};
const Loss loss = Loss::L2;

$half prediction_loss($half prediction, $half target) {
    switch (loss) {
        case Loss::L2: return (prediction - target) * (prediction - target);
        case Loss::RelativeL2: return (prediction - target) * (prediction - target) / (prediction * prediction + half(0.01));
    }
}

$half loss_gradient($half prediction, $half target, $uint batch_size) {
    switch (loss) {
        case Loss::L2: return 2 * (prediction - target);
        case Loss::RelativeL2: return half(2) * (prediction - target) / (prediction * prediction + half(0.01));
        break;
    }
}

float loss_gradient_host(float prediction, float target, uint batch_size) {
    switch (loss) {
        case Loss::L2: return 2 * (prediction - target);
        case Loss::RelativeL2: return 2 * (prediction - target) / (prediction * prediction + 0.01);
        break;
    }
}

namespace optimizer {

float learning_rate = 0.01;
float beta1 = 0.9;
float beta2 = 0.99;
float eps = 1e-8;
uint t = 0;

Buffer<float4> weights_fp;
Buffer<float4> mt;
Buffer<float4> vt;

Kernel1D optimize_kernel = []($buffer<half4> weights, $buffer<half4> gradients, $uint t, $buffer<float4> weights_fp, $buffer<float4> mt, $buffer<float4> vt) {
    $uint tid = $dispatch_x;
    $float4 w = weights_fp.read(tid);
    $half4 g = gradients.read(tid);
    $float4 m = mt.read(tid);
    $float4 v = vt.read(tid);
    $float tg;
    for (int i = 0; i < 4; i++) {
        tg = g[i];
        m[i] = beta1*m[i] + (1 - beta1)*tg;
        v[i] = beta2*v[i] + (1 - beta2)*tg*tg;
    }
    mt.write(tid, m);
    vt.write(tid, v);
    
    for (int i = 0; i < 4; i++) {
        m[i] = m[i] / (1 - pow(beta1, t.cast<float>()));
        v[i] = v[i] / (1 - pow(beta2, t.cast<float>()));

        w[i] = w[i] - learning_rate*m[i] / (sqrt(v[i]) + eps);
    }
    weights_fp.write(tid, w);
    $half4 tw = w;
    weights.write(tid, tw);

    // $float4 w = weights_fp.read(tid);
    // $half4 g = gradients.read(tid);
    // for (int i = 0; i < 4; i++) {
    //     w[i] -= learning_rate * g[i].cast<float>();
    // }
    // weights_fp.write(tid, w);
    // $half4 tw = w;
    // weights.write(tid, tw);
};
Shader1D<Buffer<half4>, Buffer<half4>, uint, Buffer<float4>, Buffer<float4>, Buffer<float4>> optimize_shader;

void init_shader();
void init_buffer(Buffer<half4> &weights);
void init(Buffer<half4> &weights) {
    init_shader();
    init_buffer(weights);
}

void optimize(Buffer<half4> &weights, Buffer<half4> &gradients) {
    t++;
    lc::stream << optimize_shader(weights, gradients, t, weights_fp, mt, vt).dispatch(weights.size());
}

}

#if use_input_encode && !use_instant_ngp
namespace encoder {
    const uint encode_width = 32;
    const uint L = encode_width / input_dim;

    Kernel1D encode_kernel = []($uint batch_size, $buffer<half4> input, $buffer<half4> fwd_tmp) {
        set_block_size(256);
        $shared<half4> smem{512};
        $uint tid = $dispatch_x % 256;
        $uint bid = $dispatch_x / 256;

        $half4 in = input.read(tid + bid*256);
        
        for (int l = 0; l < L / 2; l++) {
            sync_block();
            for (int i = 0; i < input_dim; i++) {
                for (int t = 0; t < 2; t++) {
                    $float x = in[i + t*2];
                    smem[tid/2 + i*256][t + tid%2*2] = sin(x * (1 << l) * pi);
                    smem[tid/2 + 128 + i*256][t + tid%2*2] = cos(x * (1 << l) * pi);
                    // smem[tid/2 + i*256][t + tid%2*2] = sin(x * (1 << (l*2)) * pi);
                    // smem[tid/2 + 128 + i*256][t + tid%2*2] = sin(x * (1 << (l*2+1)) * pi);
                }
            }
            sync_block();
            fwd_tmp.write(tid%128 + bid*128 + (tid/128 + l*2)*batch_size/4, smem[tid]);
            fwd_tmp.write(tid%128 + bid*128 + (tid/128 + l*2 + L)*batch_size/4, smem[tid + 256]);
        }
    };
    Shader1D<uint, Buffer<half4>, Buffer<half4>> encode_shader;

    void init() {
        encode_shader = lc::device.compile(encode_kernel);
    }
    void encode(uint batch_size, Buffer<half4> &input, Buffer<half4> &fwd_tmp) {
        lc::stream << encode_shader(batch_size, input, fwd_tmp).dispatch(batch_size / 2);
    }
}
#endif

#if use_instant_ngp
namespace encoder {
    const uint encode_width = 32;
    const uint L = 16;
    const uint T = 1 << 18;
    const uint F = 2;
    const uint N_min = 16;
    // const uint N_max = 512;
    // const float b = exp((log(N_max) - log(N_min)) / (L - 1));

    BufferView<half2> feature_table;
    BufferView<half2> feature_gradient;
    Buffer<float> feature_gradient_fp;

    BufferView<half2> ema_feature_table;

    uint level_offset[L + 1];
    Buffer<uint> level_offset_buffer;

    void init_level_offset() {
        uint offset = 0;
        for (uint i = 0; i < L; i++) {
            uint res = (1u << i) * N_min;
            uint level_size = min(res * res, T);
            level_offset[i] = offset;
            offset += level_size;
        }
        level_offset[L] = offset;
    }
    uint table_size() {
        init_level_offset();
        return level_offset[L];
    }

    Kernel1D init_feature_kernel = []($buffer<half2> feature_table) {
        set_block_size(256);
        $uint tid = $dispatch_x;
        $half2 tmp;
        tmp.x = 1.0f * as_uniform(tea(tid, 233));
        tmp.y = 1.0f * as_uniform(tea(tid, 233 + 114514));
        feature_table.write(tid, tmp);
    };
    Shader1D<Buffer<half2>> init_feature_shader;

    void calc_pos($float2 &in, $float2 &pos, $uint2 &grid_idx, $uint &grid_res) {
        $float2 tmp = in * (grid_res - 1).cast<float>();
        grid_idx = floor(tmp);
        pos = tmp - floor(tmp);
    }
    $uint table_idx($uint2 &grid_idx, $uint &grid_res, $uint &level_size) {
        $uint idx;
        $if (grid_res * grid_res <= level_size) {
            idx = grid_idx.x + grid_idx.y * grid_res;
        } $else {
            uint prime[3] = {1u, 2654435761u, 805459861u};
            idx = (grid_idx.x * prime[0]) ^ (grid_idx.y * prime[1]);
        };
        idx %= level_size;
        return idx;
    }

    Kernel2D encode_kernel = []($uint batch_size, $buffer<half2> input, $buffer<half4> fwd_tmp, $buffer<half2> feature_table, $buffer<uint> level_offsets) {
        set_block_size(512, 1);
        $shared<half4> smem{256};
        $uint tid = $dispatch_x % 512;
        $uint bid = $dispatch_x / 512;
        $uint level = $dispatch_y;
        $uint level_offset = level_offsets.read(level);
        $uint level_size = level_offsets.read(level + 1) - level_offset;

        $uint grid_res = (1u << level) * N_min;

        $float2 in = input.read(tid + bid*512);
        $float2 pos;
        $uint2 grid_idx;
        calc_pos(in, pos, grid_idx, grid_res);

        $float2 res_feature;
        for (int t = 0; t < 4; t++) {
            $float w = 1;
            $uint2 idx;
            for (int i = 0; i < 2; i++) {
                if ((t & (1 << i)) == 0) {
                    w *= 1 - pos[i];
                    idx[i] = grid_idx[i];
                }
                else {
                    w *= pos[i];
                    idx[i] = grid_idx[i] + 1;
                }
            }
            $float2 feature = feature_table.read(level_offset + table_idx(idx, grid_res, level_size));
            res_feature += w * feature;
        }
        smem[tid/4][tid%4] = res_feature.x;
        smem[tid/4 + 128][tid%4] = res_feature.y;
        sync_block();
        $if (tid < 256) {
            fwd_tmp.write(tid%128 + bid*128 + (level*2 + tid/128)*batch_size/4, smem[tid]);
        };
    };

    Kernel2D calc_feature_gradient_kernel = []($uint batch_size, $buffer<half2> input, $buffer<half4> input_grad, $buffer<float> feature_gradient_fp, $buffer<uint> level_offsets) {
        set_block_size(256, 1);
        $shared<half4> smem{128};
        $uint tid = $dispatch_x % 256;
        $uint bid = $dispatch_x / 256;

        $uint level = $dispatch_y;
        $uint level_offset = level_offsets.read(level);
        $uint level_size = level_offsets.read(level + 1) - level_offset;

        $uint grid_res = (1u << level) * N_min;

        $if (tid < 128) {
            smem[tid] = input_grad.read(tid%64 + bid*64 + (level*2 + tid/64)*batch_size/4);
        };
        sync_block();

        $float2 grad;
        grad.x = smem[tid/4][tid%4];
        grad.y = smem[tid/4 + 64][tid%4];

        $float2 in = input.read(tid + bid*256);
        $float2 pos;
        $uint2 grid_idx;
        calc_pos(in, pos, grid_idx, grid_res);

        for (int t = 0; t < 4; t++) {
            $float w = 1;
            $uint2 idx;
            for (int i = 0; i < 2; i++) {
                if ((t & (1 << i)) == 0) {
                    w *= 1 - pos[i];
                    idx[i] = grid_idx[i];
                }
                else {
                    w *= pos[i];
                    idx[i] = grid_idx[i] + 1;
                }
            }
            $uint tidx = level_offset + table_idx(idx, grid_res, level_size);
            // feature_gradient_fp.write(tidx*2, grad.x);
            // feature_gradient_fp.write(tidx*2 + 1, grad.y);
            feature_gradient_fp.write(tidx*2, grad.x * w);
            feature_gradient_fp.write(tidx*2 + 1, grad.y * w);
            // feature_gradient_fp.atomic(tidx*2).fetch_add(grad.x * w);
            // feature_gradient_fp.atomic(tidx*2 + 1).fetch_add(grad.y * w);
        }
    };

    Shader2D<uint, Buffer<half2>, Buffer<half4>, Buffer<half2>, Buffer<uint>> encode_shader;
    Shader2D<uint, Buffer<half2>, Buffer<half4>, Buffer<float>, Buffer<uint>> calc_feature_gradient_shader;

    Kernel1D clear_gradient_kernel = []($buffer<float> feature_gradient_fp) {
        $uint tid = $dispatch_x;
        feature_gradient_fp.write(tid, 0.0f);
    };
    Kernel1D copy_gradient_kernel = []($buffer<half2> feature_gradient, $buffer<float2> feature_gradient_fp) {
        $uint tid = $dispatch_x;
        $half2 g = feature_gradient_fp.read(tid);
        feature_gradient.write(tid, g);
    };
    Shader1D<Buffer<float>> clear_gradient_shader;
    Shader1D<Buffer<half2>, Buffer<float2>> copy_gradient_shader;

    void init() {
        init_feature_shader = lc::device.compile(init_feature_kernel);
        encode_shader = lc::device.compile(encode_kernel);
        calc_feature_gradient_shader = lc::device.compile(calc_feature_gradient_kernel);
        clear_gradient_shader = lc::device.compile(clear_gradient_kernel);
        copy_gradient_shader = lc::device.compile(copy_gradient_kernel);

        lc::stream << init_feature_shader(feature_table).dispatch(feature_table.size());

        level_offset_buffer = lc::device.create_buffer<uint>(L + 1);
        lc::stream << level_offset_buffer.copy_from(level_offset);
        
        feature_gradient_fp = lc::device.create_buffer<float>(table_size() * F);
    }

    void encode(uint batch_size, Buffer<half4> &input, Buffer<half4> &fwd_tmp) {
        lc::stream << encode_shader(batch_size, input.view().as<half2>(), fwd_tmp, feature_table, level_offset_buffer).dispatch(batch_size, L);

        // lc::stream.synchronize();
        // vector<half> tmp(32*16384);
        // lc::stream << fwd_tmp.copy_to(tmp.data()) << synchronize();
        // int i = 0;
        // for (float x: tmp) {
        //     print("{}: {}\n", i, x);
        //     i++;
        //     if (i > 32) break;
        // }
        
    }
    void inference_encode(uint batch_size, Buffer<half4> &input, Buffer<half4> &fwd_tmp) {
        lc::stream << encode_shader(batch_size, input.view().as<half2>(), fwd_tmp, ema_feature_table, level_offset_buffer).dispatch(batch_size, L);
    }
    void calc_feature_gradient(uint batch_size, Buffer<half4> &input, Buffer<half4> &input_grad) {
        lc::stream << clear_gradient_shader(feature_gradient_fp).dispatch(feature_gradient_fp.size())
            << calc_feature_gradient_shader(batch_size, input.view().as<half2>(), input_grad, feature_gradient_fp, level_offset_buffer).dispatch(batch_size, L)
            << copy_gradient_shader(feature_gradient, feature_gradient_fp.view().as<float2>()).dispatch(feature_gradient.size());
    }
}
#endif

namespace network {
    const uint layer_width = 32;
    const uint hidden_layers = 2;

    Buffer<half4> weight_buffer;
    BufferView<half4> layer_weight[hidden_layers - 1];
    BufferView<half4> input_weight;
    BufferView<half4> output_weight;

    Buffer<half4> gradient_buffer;
    BufferView<half4> layer_weight_gradient[hidden_layers - 1];
    BufferView<half4> input_weight_gradient;
    BufferView<half4> output_weight_gradient;

    Buffer<half4> ema_weight_buffer;
    BufferView<half4> ema_layer_weight[hidden_layers - 1];
    BufferView<half4> ema_input_weight;
    BufferView<half4> ema_output_weight;
    uint step = 0;

    void init_buffer();
    void init_shader();
    void init_weight();
    void init() {
        init_buffer();
        init_shader();
        init_weight();

#if use_input_encode
        encoder::init();
#endif
        optimizer::init(weight_buffer);
// #if use_ema_weight
//         lc::stream << ema_weight_buffer.copy_from(weight_buffer);
// #endif
    }
    void test(vector<half> &input_h, vector<half> &target_h);

    void input_forward(uint batch_size, BufferView<half4> &weight, Buffer<half4> &input, Buffer<half4> &store);
    void layer_forward(uint batch_size, BufferView<half4> &weight, Buffer<half4> &layer_act, Buffer<half4> &store);
    void output_forward(uint batch_size, BufferView<half4> &weight, Buffer<half4> &layer_act, Buffer<half4> &store);

    void layer_backward(uint batch_size, uint layer_id, Buffer<half4> &layer_grad, Buffer<half4> &fwd_act, Buffer<half4> &store);
    void output_backward(uint batch_size, Buffer<half4> &output_grad, Buffer<half4> &fwd_act, Buffer<half4> &store);

    void encode_input_backward(uint batch_size, Buffer<half4> &bwd_tmp0, Buffer<half4> &input_grad);

    void calc_output_gradiant(uint batch_size, Buffer<half4> &output, Buffer<half4> &target, Buffer<half4> &output_grad, Buffer<half4> &output_loss);

    void calc_layer_weight_gradient(uint batch_size, BufferView<half4> &weight_grad, Buffer<half4> &fwd_act, Buffer<half4> &layer_grad, Buffer<half4> &splitk_tmp);
    void calc_input_weight_gradient(uint batch_size, Buffer<half4> &input, Buffer<half4> &layer_grad, Buffer<half4> &splitk_tmp);
    void calc_output_weight_gradient(uint batch_size, Buffer<half4> &fwd_act, Buffer<half4> &ouput_grad, Buffer<half4> &splitk_tmp);

    void apply_ema_weight();

    void train_step(uint batch_size, Buffer<half4> &input, Buffer<half4> &target, Buffer<half4> &output, Buffer<half4> &output_grad, Buffer<half4> &output_loss, Buffer<half4> *fwd_tmp, Buffer<half4> *bwd_tmp, Buffer<half4> *splitk_tmp) {
        const uint n = hidden_layers;
        step++;

#if use_input_encode
        encoder::encode(batch_size, input, fwd_tmp[n]);
        layer_forward(batch_size, input_weight, fwd_tmp[n], fwd_tmp[0]);
#else
        input_forward(batch_size, input, fwd_tmp[0]);
#endif
        for (int i = 0; i < n - 1; i++) {
            layer_forward(batch_size, layer_weight[i], fwd_tmp[i], fwd_tmp[i + 1]);
        }
        output_forward(batch_size, output_weight, fwd_tmp[n - 1], output);

        calc_output_gradiant(batch_size, output, target, output_grad, output_loss);

        output_backward(batch_size, output_grad, fwd_tmp[n - 1], bwd_tmp[n - 1]);
        for (int i = n - 1; i > 0; i--) {
            layer_backward(batch_size, i - 1, bwd_tmp[i], fwd_tmp[i - 1], bwd_tmp[i - 1]);
        }
#if use_instant_ngp
        encode_input_backward(batch_size, bwd_tmp[0], bwd_tmp[n]);
#endif

#if use_input_encode
        calc_layer_weight_gradient(batch_size, input_weight_gradient, fwd_tmp[n], bwd_tmp[0], splitk_tmp[n - 1]);
#else
        calc_input_weight_gradient(batch_size, input, bwd_tmp[0], splitk_tmp[n - 1]);
#endif
#if use_instant_ngp
        encoder::calc_feature_gradient(batch_size, input, bwd_tmp[n]);
#endif
        for (int i = 0; i < n - 1; i++) {
            calc_layer_weight_gradient(batch_size, layer_weight_gradient[i], fwd_tmp[i], bwd_tmp[i + 1], splitk_tmp[i]);
        }
        calc_output_weight_gradient(batch_size, fwd_tmp[n - 1], output_grad, splitk_tmp[n]);

        optimizer::optimize(weight_buffer, gradient_buffer);
#if use_ema_weight
        apply_ema_weight();
#endif
    }

    void inference(uint batch_size, Buffer<half4> &input, Buffer<half4> &output, Buffer<half4> &fwd_tmp) {
#if use_ema_weight
    #if use_input_encode
        // encoder::encode(batch_size, input, fwd_tmp);
        encoder::inference_encode(batch_size, input, fwd_tmp);
        layer_forward(batch_size, ema_input_weight, fwd_tmp, fwd_tmp);
    #else
        input_forward(batch_size, input, fwd_tmp);
    #endif
        for (int i = 0; i < hidden_layers - 1; i++) {
            layer_forward(batch_size, ema_layer_weight[i], fwd_tmp, fwd_tmp);
        }
        output_forward(batch_size, ema_output_weight, fwd_tmp, output);
#else
    #if use_input_encode
        encoder::encode(batch_size, input, fwd_tmp);
        layer_forward(batch_size, input_weight, fwd_tmp, fwd_tmp);
    #else
        input_forward(batch_size, input, fwd_tmp);
    #endif
        for (int i = 0; i < hidden_layers - 1; i++) {
            layer_forward(batch_size, layer_weight[i], fwd_tmp, fwd_tmp);
        }
        output_forward(batch_size, output_weight, fwd_tmp, output);
#endif
    }
}

void load_image(const char* file, Image<float> &img) {
    int width = 0;
    int height = 0;
    int channel = 0;
    auto pixel = stbi_load(file, &width, &height, &channel, 4);

    // print("{}, {}, {}, {}\n", width, height, channel, (void*)pixel);

    img = lc::device.create_image<float>(PixelStorage::BYTE4, width, height);
    lc::stream << img.copy_from(pixel) << synchronize();
}

namespace trainer {
    const uint train_batch_size = 16384;
#if use_input_encode
    Buffer<half4> fwd_tmp[network::hidden_layers + 1];
#else
    Buffer<half4> fwd_tmp[network::hidden_layers];
#endif
#if use_instant_ngp
    Buffer<half4> bwd_tmp[network::hidden_layers + 1];
#else
    Buffer<half4> bwd_tmp[network::hidden_layers];
#endif
    Buffer<half4> splitk_tmp[network::hidden_layers + 1];
    Buffer<half4> output;
    Buffer<half4> output_grad;
    Buffer<half4> output_loss;
    Buffer<half4> input;
    Buffer<half4> target;

    Image<float> train_image;
    BindlessArray heap;

    Image<float> inference_image;
    Buffer<half4> inference_input;
    Buffer<half4> inference_output;
    Buffer<half4> inference_tmp;

    vector<half> tmp(output_dim*train_batch_size);

    void init_shader();
    void init_buffer();
    void init(const char* file) {
        load_image(file, train_image);
        // print("{}, {}\n", train_image.size().x, train_image.size().y);
        heap = lc::device.create_bindless_array(1);
        heap.emplace_on_update(0, train_image, Sampler::linear_linear_mirror());
        lc::stream << heap.update() <<synchronize();

        init_shader();
        init_buffer();
        // print("init trainer ok\n");

        network::init();
    }
    void prepare_train_data();
    void train_step() {
        prepare_train_data();
        network::train_step(train_batch_size, input, target, output, output_grad, output_loss, fwd_tmp, bwd_tmp, splitk_tmp);

        lc::stream << output_loss.copy_to(tmp.data()) << synchronize();
        static uint step = 0;
        float x = 0;
        for (float y: tmp) x += y;
        print("step: {}, loss: {}\n", step, x / train_batch_size);
        step++;
    }
    void inference();

    void test() {
        prepare_train_data();
        vector<half> input_h(input_dim * train_batch_size);
        vector<half> target_h(output_dim_pad * train_batch_size);

        lc::stream.synchronize();
        // print("1\n");
        lc::stream << input.copy_to(input_h.data()) << target.copy_to(target_h.data()) << synchronize();

        // print("train data:\n");
        // for (int i = 0; i < 32; i++) {
        //     print("({}, {}) -> ({}, {}, {})\n", input_h[i*2], input_h[i*2+1], target_h[i*4], target_h[i*4+1], target_h[i*4+2]);
        // }
        // print("\n");

        network::test(input_h, target_h);

        lc::stream << output_loss.copy_to(tmp.data()) << synchronize();
        float x = 0;
        for (float y: tmp) x += y;
        print("loss: {}\n", x / train_batch_size);
    }
}

namespace network {

void test(vector<half> &input_h, vector<half> &target_h) {
    const uint n = hidden_layers;
    const uint batch_size = trainer::train_batch_size;

    vector<half> layer_weight_h[hidden_layers - 1];
    for (auto &v : layer_weight_h) v = vector<half>(layer_width*layer_width);
    vector<half> input_weight_h(input_dim*layer_width);
    vector<half> output_weight_h(output_dim_pad*layer_width);

    for (int i = 0; i < hidden_layers - 1; i++) {
        lc::stream << layer_weight[i].copy_to(layer_weight_h[i].data());
    }
    lc::stream << input_weight.copy_to(input_weight_h.data());
    lc::stream << output_weight.copy_to(output_weight_h.data());
    lc::stream.synchronize();

    // for (int i = 0; i < hidden_layers - 1; i++) {
    //     print("layer weight {}:\n", i);
    //     print("[");
    //     for (float x : layer_weight_h[i]) {
    //         print("{}, ", x);
    //     }
    //     print("]\n\n");
    // }
    // print("input weight:\n");
    // print("[");
    // for (float x : input_weight_h) {
    //     print("{}, ", x);
    // }
    // print("]\n\n");
    // print("output weight:\n");
    // print("[");
    // for (float x : output_weight_h) {
    //     print("{}, ", x);
    // }
    // print("]\n\n");

    train_step(batch_size, trainer::input, trainer::target, trainer::output, trainer::output_grad, trainer::output_loss, trainer::fwd_tmp, trainer::bwd_tmp, trainer::splitk_tmp);

    vector<float> fwd_tmp_h[hidden_layers], bwd_tmp_h[hidden_layers];
    for (auto &v : fwd_tmp_h) v = vector<float>(layer_width * batch_size);
    for (auto &v : bwd_tmp_h) v = vector<float>(layer_width * batch_size);
    vector<float> output_h = vector<float>(output_dim_pad * batch_size);
    vector<float> output_grad_h = vector<float>(output_dim * batch_size);
    vector<float> layer_weight_gradient_h[hidden_layers - 1];
    for (auto &v : layer_weight_gradient_h) v = vector<float>(layer_width*layer_width);
    vector<float> input_weight_gradient_h(layer_width*input_dim);
    vector<float> output_weight_gradient_h(layer_width*output_dim);

    auto input_forward_h = [](vector<half> &a, vector<half> &b, vector<float> &c) {
        for (int k = 0; k < input_dim; k++) {
            for (int x = 0; x < layer_width; x++) {
                float tmp = a[x + k*layer_width];
                for (int y = 0; y < batch_size; y++) {
                    c[y + x*batch_size] += tmp * b[k + y*input_dim];
                }
            }
        }
        for (float &x : c) {
            activation_forward_host(x);
        }
    };
    auto layer_forward_h = [](vector<half> &a, vector<float> &b, vector<float> &c) {
        for (int k = 0; k < layer_width; k++) {
            for (int x = 0; x < layer_width; x++) {
                float tmp = a[x + k*layer_width];
                for (int y = 0; y < batch_size; y++) {
                    c[y + x*batch_size] += tmp * b[y + k*batch_size];
                }
            }
        }
        for (float &x : c) {
            activation_forward_host(x);
        }
    };
    auto output_forward_h = [](vector<half> &a, vector<float> &b, vector<float> &c) {
        for (int x = 0; x < output_dim; x++) {
            for (int k = 0; k < layer_width; k++) {
                float tmp = a[k + x*layer_width];
                for (int y = 0; y < batch_size; y++) {
                    c[x + y*output_dim_pad] += tmp * b[y + k*batch_size];
                }
            }
        }
    };
    auto calc_output_gradiant_h = [](vector<float> &output, vector<half> &target, vector <float> &store) {
        for (int x = 0; x < output_dim; x++) {
            for (int y = 0; y < batch_size; y++) {
                store[y + x*batch_size] = loss_gradient_host(output[x + y*output_dim_pad], target[x + y*output_dim_pad], batch_size);
            }
        }
    };
    auto output_backward_h = [](vector<half> &a, vector<float> &b, vector<float> &fwd_act_h, vector<float> &c) {
        for (int x = 0; x < layer_width; x++) {
            for (int k = 0; k < output_dim; k++) {
                float tmp = a[x + k*layer_width];
                for (int y = 0; y < batch_size; y++) {
                    c[y + x*batch_size] += tmp * b[y + k*batch_size];
                }
            }
        }
        int i = 0;
        for (float &x : c) {
            activation_backward_host(x, fwd_act_h[i]);
            i++;
        }
    };
    auto layer_backward_h = [](vector<half> &a, vector<float> &b, vector<float> &fwd_act_h, vector<float> &c) {
        for (int x = 0; x < layer_width; x++) {
            for (int k = 0; k < layer_width; k++) {
                float tmp = a[k + x*layer_width];
                for (int y = 0; y < batch_size; y++) {
                    c[y + x*batch_size] += tmp * b[y + k*batch_size];
                }
            }
        }
        int i = 0;
        for (float &x : c) {
            activation_backward_host(x, fwd_act_h[i]);
            i++;
        }
    };
    auto calc_layer_weight_gradient_h = [](vector<float> &fwd_tmp, vector<float> &bwd_tmp, vector<float> &c) {
        for (int x = 0; x < layer_width; x++) {
            for (int y = 0; y < layer_width; y++) {
                for (int k = 0; k < batch_size; k++) {
                    c[x + y*layer_width] += bwd_tmp[k + x*batch_size] * fwd_tmp[k + y*batch_size];
                }
            }
        }
        for (float &x: c) x /= batch_size;
    };
    auto calc_input_weight_gradient_h = [](vector<float> &bwd_tmp, vector<half> &input, vector<float> &c) {
        for (int x = 0; x < layer_width; x++) {
            for (int y = 0; y < input_dim; y++) {
                for (int k = 0; k < batch_size; k++) {
                    c[x + y*layer_width] += bwd_tmp[k + x*batch_size] * input[y + k*input_dim];
                }
            }
        }
        for (float &x: c) x /= batch_size;
    };
    auto calc_output_weight_gradient_h = [](vector<float> &fwd_act, vector<float> &output_grad, vector<float> &c) {
        for (int x = 0; x < layer_width; x++) {
            for (int y = 0; y < output_dim; y++) {
                for (int k = 0; k < batch_size; k++) {
                    c[x + y*layer_width] += fwd_act[k + x*batch_size] * output_grad[k + y*batch_size];
                }
            }
        }
        for (float &x: c) x /= batch_size;
    };

    input_forward_h(input_weight_h, input_h, fwd_tmp_h[0]);
    for (int i = 0; i < n - 1; i++) {
        layer_forward_h(layer_weight_h[i], fwd_tmp_h[i], fwd_tmp_h[i+1]);
    }
    output_forward_h(output_weight_h, fwd_tmp_h[n-1], output_h);

    calc_output_gradiant_h(output_h, target_h, output_grad_h);

    output_backward_h(output_weight_h, output_grad_h, fwd_tmp_h[n-1], bwd_tmp_h[n-1]);
    for (int i = n - 1; i > 0; i--) {
        layer_backward_h(layer_weight_h[i - 1], bwd_tmp_h[i], fwd_tmp_h[i - 1], bwd_tmp_h[i - 1]);
    }

    calc_input_weight_gradient_h(bwd_tmp_h[0], input_h, input_weight_gradient_h);
    for (int i = 0; i < n - 1; i++) {
        calc_layer_weight_gradient_h(fwd_tmp_h[i], bwd_tmp_h[i + 1], layer_weight_gradient_h[i]);
    }
    calc_output_weight_gradient_h(fwd_tmp_h[n - 1], output_grad_h, output_weight_gradient_h);

    vector<half> tmp(1e7);
    lc::stream.synchronize();

    auto compare_buffer = [&](vector<float> &buf, string name) {
        print("{}_h: [", name);
        for (int i = 0; i < 64; i++) {
            print("{}", buf[i]);
            if (i < 63) print(", ");
        }
        print("]\n");
        print("{}_d: [", name);
        for (int i = 0; i < 64; i++) {
            print("{}", tmp[i]);
            if (i < 63) print(", ");
        }
        print("]\n");
        float f_err = 0;
        float f_err2 = 0;
        int err_c = 0;
        int i = 0;
        float M = -1e9, m = 1e9;
        for (float x: buf) {
            M = max(M, x);
            m = min(m, x);
            float y = tmp[i];
            float err = abs(y - x) / max(abs(x), abs(y));
            float err2 = abs(y - x);
            if (err > f_err) {
                print("!inc error {}: {}, {}; f_err: {}\n", i, x, y, err);
            }
            f_err = max(f_err, err);
            f_err2 = max(f_err2, err2);
            if (err > 0.01 || err2 > 0.005) {
                if (err_c < 32) {
                    print("error {}: {}, {}\n", i, x, y);
                }
                err_c++;
            }
            i++;
        }
        print("max: {}, min: {}\n", M, m);
        print("f_err: {}\n", f_err);
        print("f_err2: {}\n", f_err2);
        print("err_c: {}\n", err_c);
        print("ok\n\n");
    };

    for (int i = 0; i < n; i++) {
        lc::stream << trainer::fwd_tmp[i].copy_to(tmp.data()) << synchronize();
        compare_buffer(fwd_tmp_h[i], format("fwd_tmp {}", i));
    }

    lc::stream << trainer::output.copy_to(tmp.data()) << synchronize();
    compare_buffer(output_h, "output");

    lc::stream << trainer::output_grad.copy_to(tmp.data()) << synchronize();
    compare_buffer(output_grad_h, "output_grad");

    for (int i = 0; i < n; i++) {
        lc::stream << trainer::bwd_tmp[i].copy_to(tmp.data()) << synchronize();
        compare_buffer(bwd_tmp_h[i], format("bwd_tmp {}", i));
    }

    for (int i = 0; i < n-1; i++) {
        lc::stream << layer_weight_gradient[i].copy_to(tmp.data()) << synchronize();
        compare_buffer(layer_weight_gradient_h[i], format("layer_weight_gradient {}", i));
    }

    lc::stream << input_weight_gradient.copy_to(tmp.data()) << synchronize();
    compare_buffer(input_weight_gradient_h, "input_weight_gradient");

    lc::stream << output_weight_gradient.copy_to(tmp.data()) << synchronize();
    compare_buffer(output_weight_gradient_h, "output_weight_gradient");

    print("\n");

    // input_forward(batch_size, input, fwd_tmp[0]);
    // for (int i = 0; i < n - 1; i++) {
    //     layer_forward(batch_size, i, fwd_tmp[i], fwd_tmp[i + 1]);
    // }
    // output_forward(batch_size, fwd_tmp[n - 1], output);
    // calc_output_gradiant(batch_size, output, target, output_grad);
    // output_backward(batch_size, output_grad, fwd_tmp[n - 1], bwd_tmp[n - 1]);
    // for (int i = n - 1; i > 0; i--) {
    //     layer_backward(batch_size, i - 1, bwd_tmp[i], fwd_tmp[i - 1], bwd_tmp[i - 1]);
    // }
    // calc_input_weight_gradient(batch_size, input, bwd_tmp[0]);
    // for (int i = 0; i < n - 1; i++) {
    //     calc_layer_weight_gradient(batch_size, i, fwd_tmp[i], bwd_tmp[i + 1]);
    // }
    // calc_output_weight_gradient(batch_size, fwd_tmp[n - 1], output_grad);
}

}

template<class T>
void print(vector<T> &v, string name) {
    fmt::print("{}: [", name);
    for (int i = 0; i < 32; i++) {
        float x = v[i];
        fmt::print("{}, ", x);
    }
    fmt::print("]\n");
}

int main(int argc, char *argv[]) {
    lc::init(argv[0]);
    // auto img = load_image("assets/nahida.jpeg");
    trainer::init("assets/nahida.jpeg");
    // trainer::test();

    vector<float> tmp;
    tmp.push_back(1);

    // const float a = 0.99;
    // for (int i = 1; i < 10; i++) {
    //     float t = (1 - a) / (1 - pow(a, i));
    //     for (float &x : tmp) x *= (1 - t);
    //     tmp.push_back(t);

    //     print("step {}\n", i);
    //     for (float &x : tmp) print("{} ", x);
    //     print("\n");
    // }

    // const uint n = network::weight_buffer.size() * 4;
    // vector<half> weight(n);
    // vector<float> ema_weight(n);
    // vector<half> tmp(n);

    // lc::stream << network::weight_buffer.copy_to(weight.data())
    //     << network::ema_weight_buffer.copy_to(tmp.data())
    //     << synchronize();

    // print(weight, "weight");
    // print(ema_weight, "ema_weight_h");
    // print(tmp, "ema_weight_d");

    // const float a = 0.99;

    // for (int t = 0; t < 10; t++) {
    //     trainer::train_step();
    //     print("step {}\n", network::step);
    //     lc::stream << network::weight_buffer.copy_to(weight.data())
    //         << network::ema_weight_buffer.copy_to(tmp.data())
    //         << synchronize();
        
    //     int i = 0;
    //     for (float &x : ema_weight) {
    //         float w = weight[i];
    //         float t = (1-a)/(1-pow(a,network::step));
    //         x = t*w + (1-t)*x;
    //         // x = (1-a)/(1-pow(a,network::step))*w + a*(1-pow(a,network::step-1))*x;
    //         i++;
    //     }

    //     print(weight, "weight");
    //     print(ema_weight, "ema_weight_h");
    //     print(tmp, "ema_weight_d");
    //     print("\n");
    // }

    Window window{"", 1920, 1080};
    Swapchain swapchain = lc::device.create_swapchain(
        window.native_handle(),
        lc::stream,
        make_uint2(1920, 1080),
        false, false,
        3
    );

    while(!window.should_close()){
        trainer::train_step();
        // break;
    //     // trainer::test();
        trainer::inference();
        lc::stream << swapchain.present(trainer::inference_image);
        // lc::stream << swapchain.present(trainer::train_image);
        window.poll_events();
    }
    lc::stream.synchronize();
    
    // print("{}\n", encoder::table_size());

    return 0;
}

// func impl

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

namespace network {

void load_32x2($buffer<half4> &buf, $shared<half4> &smem) {
    $uint tid = $dispatch_x % 256;
    $if (tid < 16) {
        smem[tid] = buf.read(tid);
    };
}
void load_transpose_2x32($buffer<half4> &buf, $shared<half4> &smem) {
    $uint tid = $dispatch_x % 256;
    $if (tid < 16) {
        $half4 tmp = buf.read(tid);
        for (int i = 0; i < 4; i++) {
            smem[tid%8*4 + i][tid/8] = tmp[i];
        }
    };
}
void load_32x3($buffer<half4> &buf, $shared<half4> &smem) {
    $uint tid = $dispatch_x % 256;
    $if (tid < 24) {
        smem[tid] = buf.read(tid);
    };
}
void load_transpose_3x32($buffer<half4> &buf, $shared<half4> &smem) {
    $uint tid = $dispatch_x % 256;
    $if (tid < 24) {
        $half4 tmp = buf.read(tid);
        for (int i = 0; i < 4; i++) {
            smem[tid%8*4 + i][tid/8] = tmp[i];
        }
    };
}
void load_32x32($buffer<half4> &buf, $shared<half4> &smem) {
    $uint tid = $dispatch_x % 256;
    smem[tid] = buf.read(tid);
}
void load_transpose_32x32($buffer<half4> &buf, $shared<half4> &smem) {
    $uint tid = $dispatch_x % 256;
    $half4 tmp = buf.read(tid);
    for (int i = 0; i < 4; i++) {
        smem[tid/32 + (tid%8*4 + i)*8][tid/8%4] = tmp[i];
    }
}
void load_transpose_512x2($buffer<half4> &buf, $shared<half4> &smem, $uint offset) {
    $uint tid = $dispatch_x % 256;
    $half4 tmp = buf.read(tid + offset);
    for (int i = 0; i < 4; i++) {
        smem[tid/2 + i%2*128][i/2 + tid%2*2] = tmp[i];
    }
}
void load_512x3($buffer<half4> &buf, $shared<half4> &smem, $uint offset, $uint stride) {
    $uint tid = $dispatch_x % 256;
    smem[tid] = buf.read(offset + tid/128*stride);
    $if (tid < 128) {
        smem[tid + 256] = buf.read(offset + (2 + tid/128)*stride);
    };
}
void load_512x4($buffer<half4> &buf, $shared<half4> &smem, $uint offset, $uint stride) {
    $uint tid = $dispatch_x % 256;
    smem[tid] = buf.read(offset + tid/128*stride);
    smem[tid + 256] = buf.read(offset + (2 + tid/128)*stride);
}

void out_product_4x4_r($half4 &a, $half4 &b, $half4 *acc) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            acc[i][j] += a[i] * b[j];
        }
    }
}
void out_product_4x4_c($half4 &a, $half4 &b, $half4 *acc) {
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            acc[j][i] += a[i] * b[j];
        }
    }
}
void out_product_3x4_c($half4 &a, $half4 &b, $half4 *acc) {
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 3; i++) {
            acc[j][i] += a[i] * b[j];
        }
    }
}
void out_product_2x4_r($half4 &a, $half4 &b, $half4 *acc) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            acc[i][j] += a[i] * b[j];
        }
    }
}
void out_product_8x8_r($half4 *a, $half4 *b, $half4 *acc) {
    for (int tx = 0; tx < 2; tx++) {
        for (int ty = 0; ty < 2; ty++) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    acc[i + (ty + tx*2)*4][j] += a[tx][i] * b[ty][j];
                }
            }
        }
    }
}

Kernel1D output_forward_kernel = []($uint batch_size, $buffer<half4> weight, $buffer<half4> act, $buffer<half4> store) {
    set_block_size(256);
    $shared<half4> a_tile{32};
    $half4 acc[4];
    $half4 a_frag;
    $half4 b_frag;

    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;

    load_transpose_3x32(weight, a_tile);
    sync_block();
    $for (k, layer_width) {
        a_frag = a_tile[k];
        b_frag = act.read(tid + bid*256 + k*batch_size/4);
        out_product_3x4_c(a_frag, b_frag, acc);
    };
    for (int i = 0; i < 4; i++) {
        store.write(i + tid*4 + bid*256*4, acc[i]);
    }
};

// 放弃挣扎
void store_res_tile($buffer<half4> &buf, $shared<half4> &smem, $half4 *acc, $uint stride, $uint tid, $uint bid) {
    // $uint tid = $dispatch_x % 256;
    // $uint bid = $dispatch_x / 256;
    $uint x_ofs = tid%32/8;
    $uint y_ofs = tid%8 + tid/32*16;
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 4; i++) {
            sync_block();
            smem[y_ofs + x_ofs*128] = acc[i + t*2*4];
            smem[y_ofs + 8 + x_ofs*128] = acc[i + (t*2 + 1)*4];
            sync_block();

            buf.write(tid%128 + bid*128 + (i + t*16 + tid/128*4)*stride/4, smem[tid]);
            buf.write(tid%128 + bid*128 + (i + t*16 + (tid/128 + 2)*4)*stride/4, smem[tid + 256]);
        }
    }
}

// 放弃挣扎
void apply_activation_backward_tile($buffer<half4> &buf, $shared<half4> &smem, $buffer<half4> &fwd_act, $half4 *acc, $uint stride, $uint tid, $uint bid) {
    // $uint tid = $dispatch_x % 256;
    // $uint bid = $dispatch_x / 256;
    $uint x_ofs = tid%32/8;
    $uint y_ofs = tid%8 + tid/32*16;
    $half4 tmp;
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 4; i++) {
            sync_block();
            smem[tid] = fwd_act.read(tid%128 + bid*128 + (i + t*16 + tid/128*4)*stride/4);
            smem[tid + 256] = fwd_act.read(tid%128 + bid*128 + (i + t*16 + (tid/128 + 2)*4)*stride/4);
            sync_block();
            tmp = smem[y_ofs + x_ofs*128];
            apply_activation_backward(acc[i + t*2*4], tmp);
            tmp = smem[y_ofs + 8 + x_ofs*128];
            apply_activation_backward(acc[i + (t*2 + 1)*4], tmp);
        }
    }
}

Kernel1D output_backward_kernel = []($uint batch_size, $buffer<half4> weight, $buffer<half4> grad, $buffer<half4> fwd_act, $buffer<half4> store) {
    set_block_size(256);
    $shared<half4> a_tile{24};
    $shared<half4> b_tile{512};
    $half4 acc[16];
    $half4 a_frag[2];
    $half4 b_frag[2];

    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;

    $uint x_ofs = tid%32/8;
    $uint y_ofs = tid%8 + tid/32*16;

    load_32x3(weight, a_tile);
    load_512x3(grad, b_tile, tid%128 + bid*128, batch_size/4);
    sync_block();

    $for (k, output_dim) {
        for (int t = 0; t < 2; t++) {
            a_frag[t] = a_tile[x_ofs + t*4 + k*8];
            b_frag[t] = b_tile[y_ofs + t*8 + k*128];
        }
        out_product_8x8_r(a_frag, b_frag, acc);
    };
    apply_activation_backward_tile(store, b_tile, fwd_act, acc, batch_size, tid, bid);
    store_res_tile(store, b_tile, acc, batch_size, tid, bid);
};

void layer_kernel_impl($uint batch_size, $buffer<half4> &weight, $buffer<half4> &buf, $buffer<half4> &store, bool is_backward, bool use_act, $buffer<half4> *fwd_act) {
    set_block_size(256);
    $shared<half4> a_tile{256};
    $shared<half4> b_tile{512};
    $half4 acc[16];
    $half4 a_frag[2];
    $half4 b_frag[2];

    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;

    $uint x_ofs = tid%32/8;
    $uint y_ofs = tid%8 + tid/32*16;

    if (!is_backward) load_32x32(weight, a_tile);
    else load_transpose_32x32(weight, a_tile);

    $for (k, 0u, layer_width, 4u) {
        sync_block();
        load_512x4(buf, b_tile, tid%128 + bid*128 + k*batch_size/4, batch_size/4);
        sync_block();
        $for (k1, 4) {
            for (int t = 0; t < 2; t++) {
                a_frag[t] = a_tile[x_ofs + t*4 + (k + k1)*8];
                b_frag[t] = b_tile[y_ofs + t*8 + k1*128];
            }
            out_product_8x8_r(a_frag, b_frag, acc);
        };
    };
    if (use_act) {
        if(!is_backward) {
            for (int i = 0; i < 16; i++) {
                apply_activation_forward(acc[i]);
            }
        }
        else {
            apply_activation_backward_tile(store, b_tile, *fwd_act, acc, batch_size, tid, bid);
        }
    }
    store_res_tile(store, b_tile, acc, batch_size, tid, bid);
}

Kernel1D layer_forward_kernel = []($uint batch_size, $buffer<half4> weight, $buffer<half4> act, $buffer<half4> store) {
    layer_kernel_impl(batch_size, weight, act, store, false, true, nullptr);
};
Kernel1D layer_backward_kernel = []($uint batch_size, $buffer<half4> weight, $buffer<half4> grad, $buffer<half4> fwd_act, $buffer<half4> store) {
    layer_kernel_impl(batch_size, weight, grad, store, true, true, &fwd_act);
};

Kernel1D encode_input_backward_kernel = []($uint batch_size, $buffer<half4> input_weight, $buffer<half4> grad, $buffer<half4> store) {
    layer_kernel_impl(batch_size, input_weight, grad, store, true, false, nullptr);
};

Kernel1D input_forward_kernel = []($uint batch_size, $buffer<half4> weight, $buffer<half4> input, $buffer<half4> store) {
    set_block_size(256);
    $shared<half4> a_tile{16};
    $shared<half4> b_tile{512};
    $half4 acc[16];
    $half4 a_frag[2];
    $half4 b_frag[2];

    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;

    $uint x_ofs = tid%32/8;
    $uint y_ofs = tid%8 + tid/32*16;

    load_32x2(weight, a_tile);
    load_transpose_512x2(input, b_tile, bid*256);
    sync_block();

    $for (k, input_dim) {
        for (int t = 0; t < 2; t++) {
            a_frag[t] = a_tile[x_ofs + t*4 + k*8];
            b_frag[t] = b_tile[y_ofs + t*8 + k*128];
        }
        out_product_8x8_r(a_frag, b_frag, acc);
    };
    for (int i = 0; i < 16; i++) {
        apply_activation_forward(acc[i]);
    }
    store_res_tile(store, b_tile, acc, batch_size, tid, bid);
};

Kernel1D calc_output_gradient_kernel = []($uint batch_size, $buffer<half4> output, $buffer<half4> target, $buffer<half4> output_grad, $buffer<half4> output_loss) {
    set_block_size(256);
    $uint tid = $dispatch_x;
    $half4 output_f[4];
    $half4 target_f[4];
    $half4 grad;
    $half4 loss;
    for (int i = 0; i < 4; i++) {
        output_f[i] = output.read(tid*4 + i);
        target_f[i] = target.read(tid*4 + i);
    }
    for (int t = 0; t < 3; t++) {
        for (int i = 0; i < 4; i++) {
            grad[i] = loss_gradient(output_f[i][t], target_f[i][t], batch_size);
            loss[i] = prediction_loss(output_f[i][t], target_f[i][t]);
        }
        output_grad.write(tid + t*batch_size/4, grad);
        output_loss.write(tid + t*batch_size/4, loss);
    }
};

const uint tile_k = 256;

Kernel1D calc_layer_weight_gradient_kernel = []($uint batch_size, $buffer<half4> bwd_grad, $buffer<half4> fwd_act, $buffer<half4> splitk_tmp) {
    set_block_size(64);
    $shared<half4> a_tile{64};
    $shared<half4> b_tile{64};
    $half4 acc[4];
    $half4 a_frag;
    $half4 b_frag;

    $uint tid = $dispatch_x % 64;
    $uint bid = $dispatch_x / 64;

    $uint x_ofs = tid % 8;
    $uint y_ofs = tid / 8;

    $for (k, tile_k / 8) {
        sync_block();
        a_frag = bwd_grad.read(tid%2 + k*2 + bid*tile_k/4 + tid/2*batch_size/4);
        b_frag = fwd_act.read(tid%2 + k*2 + bid*tile_k/4 + tid/2*batch_size/4);
        for (int i = 0; i < 4; i++) {
            a_tile[tid/8 + (tid%2*4 + i)*8][tid/2%4] = a_frag[i];
            b_tile[tid/8 + (tid%2*4 + i)*8][tid/2%4] = b_frag[i];
        }
        sync_block();
        $for (k1, 8) {
            a_frag = a_tile[x_ofs + k1*8];
            b_frag = b_tile[y_ofs + k1*8];
            out_product_4x4_c(a_frag, b_frag, acc);
        };
    };
    for (int i = 0; i < 4; i++) {
        splitk_tmp.write(x_ofs + (y_ofs*4 + i)*8 + bid*layer_width*layer_width/4, acc[i]);
    }
};

Kernel1D calc_input_weight_gradient_kernel = []($uint batch_size, $buffer<half4> bwd_grad, $buffer<half4> input, $buffer<half4> splitk_tmp) {
    set_block_size(32);
    $shared<half4> a_tile{512};
    $shared<half4> b_tile{64};
    // $half4 acc[2];
    $array<half4, 2> acc;
    $half4 a_frag;
    $half4 b_frag;

    $uint tid = $dispatch_x % 32;
    $uint bid = $dispatch_x / 32;

    $for (k1, tile_k / 64) {
        b_frag = input.read(tid + k1*32 + bid*128);
        for (int i = 0; i < 4; i++) {
            b_tile[tid*2 + i/2][i%2] = b_frag[i];
        }
        for (int i = 0; i < 16; i++) {
            a_frag = bwd_grad.read(tid%16 + k1*16 + bid*tile_k/4 + (tid/16 + i*2)*batch_size/4);
            for (int j = 0; j < 4; j++) {
                a_tile[(tid/16 + i*2)/4 + (tid%16*4 + j)*8][(tid/16 + i*2)%4] = a_frag[j];
            }
        }
        $for (k, 64/4) {
            a_frag = a_tile[tid + k*32];
            b_frag = b_tile[tid/8 + k*4];
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 4; i++) {
                    acc[j][i] += a_frag[i] * b_frag[j];
                }
            }
        };
    };
    for (int i = 0; i < 2; i++) {
        $half4 tmp = warp_read_lane(acc[(tid/8%2)^(i^1)], tid^(8 + i*8));
        for (int i = 0; i < 4; i++) {
            acc[tid/8%2][i] += tmp[i];
        }
    }
    $if (tid < 16) {
        splitk_tmp.write(tid + bid*16, acc[tid/8]);
    };
};

Kernel1D calc_output_weight_gradient_kernel = []($uint batch_size, $buffer<half4> fwd_act, $buffer<half4> output_grad, $buffer<half4> splitk_tmp) {
    set_block_size(32);
    $shared<half4> a_tile{512};
    $shared<half4> b_tile{64};
    $array<half4, 3> acc;
    $half4 a_frag;
    $half4 b_frag;

    $uint tid = $dispatch_x % 32;
    $uint bid = $dispatch_x / 32;

    $for (k1, tile_k / 64) {
        b_frag = output_grad.read(tid%16 + k1*16 + bid*tile_k/4 + tid/16*batch_size/4);
        for (int i = 0; i < 4; i++) {
            b_tile[tid%16*4 + i][tid/16] = b_frag[i];
        }
        $if (tid < 16) {
            b_frag = output_grad.read(tid + k1*16 + bid*tile_k/4 + 2*batch_size/4);
            for (int i = 0; i < 4; i++) {
                b_tile[tid*4 + i][2] = b_frag[i];
            }
        };
        for (int i = 0; i < 16; i++) {
            a_frag = fwd_act.read(tid%16 + k1*16 + bid*tile_k/4 + (tid/16 + i*2)*batch_size/4);
            for (int j = 0; j < 4; j++) {
                a_tile[(tid/16 + i*2)/4 + (tid%16*4 + j)*8][(tid/16 + i*2)%4] = a_frag[j];
            }
        }
        $for (k, 64/4) {
            a_frag = a_tile[tid + k*32];
            b_frag = b_tile[tid/8 + k*4];
            for (int j = 0; j < 3; j++) {
                for (int i = 0; i < 4; i++) {
                    acc[j][i] += a_frag[i] * b_frag[j];
                }
            }
        };
    };
    for (int i = 0; i < 2; i++) {
        $half4 tmp = warp_read_lane(acc[(tid/8%2)^(i^1)], tid^(8 + i*8));
        for (int j = 0; j < 4; j++) {
            acc[tid/8%2][j] += tmp[j];
        }
        tmp = warp_read_lane(acc[2], tid^(8 + i*8));
        for (int j = 0; j < 4; j++) {
            acc[2][j] += tmp[j];
        }
    }
    $if (tid < 24) {
        splitk_tmp.write(tid + bid*24, acc[tid/8]);
    };
};

Kernel1D splitk_reduce_kernel = []($uint batch_size, $uint stride, $buffer<half4> splitk_tmp, $buffer<half4> store) {
    $uint tid = $dispatch_x;
    $float4 acc;
    $half4 tmp;
    $for (t, batch_size / tile_k) {
        tmp = splitk_tmp.read(tid + t*stride);
        for (int i = 0; i < 4; i++) {
            acc[i] += tmp[i].cast<float>();
        }
    };
    for (int i = 0; i < 4; i++) {
        tmp[i] = acc[i] / batch_size;
        // tmp[i] = 1;
    }
    store.write(tid, tmp);
};

Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>> output_forward_shader;
Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>> layer_forward_shader;
Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>> input_forward_shader;

Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>, Buffer<half4>> output_backward_shader;
Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>, Buffer<half4>> layer_backward_shader;

Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>> encode_input_backward_shader;

Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>, Buffer<half4>> calc_output_gradient_shader;

Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>> calc_layer_weight_gradient_shader;
Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>> calc_input_weight_gradient_shader;
Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>> calc_output_weight_gradient_shader;

Shader1D<uint, uint, Buffer<half4>, Buffer<half4>> splitk_reduce_shader;

const uint layer_weight_size = layer_width * layer_width;
#if use_input_encode
const uint input_weight_size = encoder::encode_width * layer_width;
#else
const uint input_weight_size = input_dim * layer_width;
#endif
const uint output_weight_size = output_dim * layer_width;
#if use_instant_ngp
const uint feature_table_size = encoder::table_size() * encoder::F;
#endif

const uint weight_size = 
        layer_weight_size * (hidden_layers - 1)
        + input_weight_size
        + output_weight_size
#if use_instant_ngp
        + feature_table_size
#endif
    ;

void init_buffer() {
    weight_buffer = lc::device.create_buffer<half4>(weight_size / 4);
    gradient_buffer = lc::device.create_buffer<half4>(weight_size / 4);
    ema_weight_buffer = lc::device.create_buffer<half4>(weight_size / 4);
    uint offset = 0;
    for (int i = 0; i < hidden_layers - 1; i++) {
        layer_weight[i] = weight_buffer.view(offset, layer_weight_size / 4);
        layer_weight_gradient[i] = gradient_buffer.view(offset, layer_weight_size / 4);
        ema_layer_weight[i] = ema_weight_buffer.view(offset, layer_weight_size / 4);
        offset += layer_weight_size / 4;
    }
    input_weight = weight_buffer.view(offset, input_weight_size / 4);
    input_weight_gradient = gradient_buffer.view(offset, input_weight_size / 4);
    ema_input_weight = ema_weight_buffer.view(offset, input_weight_size / 4);
    offset += input_weight_size / 4;

    output_weight = weight_buffer.view(offset, output_weight_size / 4);
    output_weight_gradient = gradient_buffer.view(offset, output_weight_size / 4);
    ema_output_weight = ema_weight_buffer.view(offset, output_weight_size / 4);
    offset += output_weight_size / 4;

#if use_instant_ngp
    encoder::feature_table = weight_buffer.view(offset, feature_table_size / 4).as<half2>();
    encoder::feature_gradient = gradient_buffer.view(offset, feature_table_size / 4).as<half2>();
    encoder::ema_feature_table = ema_weight_buffer.view(offset, feature_table_size / 4).as<half2>();
#endif
}

enum weight_type { hidden, input, output };
int fan_in(weight_type t) {
    switch (t) {
        case hidden: return layer_width;
#if use_input_encode
        case input: return encoder::encode_width;
#else
        case input: return input_dim;
#endif
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

Kernel1D init_weight_kernel = []($buffer<half4> weight_buffer) {
    set_block_size(256);
    $uint tid = $dispatch_x;

    $half4 w;
    $for (i, 4) {
        $uint s = tea(tid + 233, i + 114514);
        $float f = as_uniform(s);
        w[i] = f;
    };

    const uint ofs1 = layer_weight_size * (hidden_layers - 1) / 4;
    const uint ofs2 = ofs1 + input_weight_size / 4;
    const uint ofs3 = ofs2 + output_weight_size / 4;

    float weight_scale[3];
    if (activation == Activation::None) {
        weight_scale[0] = sqrt(6.0 / fan_in(hidden));
        weight_scale[1] = 30.0 / fan_in(input);
        weight_scale[2] = sqrt(6.0 / fan_in(output));
    }
    else {
        weight_scale[0] = sqrt(6.0 / (fan_in(hidden) + fan_out(hidden)));
        weight_scale[1] = sqrt(6.0 / (fan_in(input) + fan_out(input)));
        weight_scale[2] = sqrt(6.0 / (fan_in(output) + fan_out(output)));
    }

    $float s;
    $if (tid < ofs1) {
        s = weight_scale[0];
    } $elif (tid < ofs2) {
        s = weight_scale[1];
    } $else {
        s = weight_scale[2];
    };

    for (int i = 0; i < 4; i++) {
        w[i] = (w[i]*2 - 1) * s;
    }
    weight_buffer.write(tid, w);
};

Shader1D<Buffer<half4>> init_weight_shader;

Kernel1D apply_ema_weight_kernel = []($buffer<half4> weight_buffer, $buffer<half4> ema_weight_buffer, $uint step) {
    set_block_size(256);
    const float a = 0.99;
    $uint tid = $dispatch_x;
    $half4 w = weight_buffer.read(tid);
    $half4 ema_w = ema_weight_buffer.read(tid);
    $float t = step;
    for (int i = 0; i < 4; i++) {
        $float r = (1 - a) / (1 - pow(a, t));
        ema_w[i] = r * w[i].cast<float>() + (1 - r) * ema_w[i].cast<float>();
        // ema_w[i] = (1 - a) / (1 - pow(a, t)) * w[i].cast<float>() + a * (1 - pow(a, t-1)) * ema_w[i].cast<float>();
        // ema_w[i] = (1 - a) * w[i].cast<float>() + a * ema_w[i].cast<float>();
    }
    ema_weight_buffer.write(tid, ema_w);
};

Shader1D<Buffer<half4>, Buffer<half4>, uint> apply_ema_weight_shader;

void init_shader() {
    init_weight_shader = lc::device.compile(init_weight_kernel);

    output_forward_shader = lc::device.compile(output_forward_kernel);
    layer_forward_shader = lc::device.compile(layer_forward_kernel);
    input_forward_shader = lc::device.compile(input_forward_kernel);

    output_backward_shader = lc::device.compile(output_backward_kernel);
    layer_backward_shader = lc::device.compile(layer_backward_kernel);

    encode_input_backward_shader = lc::device.compile(encode_input_backward_kernel);

    calc_output_gradient_shader = lc::device.compile(calc_output_gradient_kernel);

    calc_layer_weight_gradient_shader = lc::device.compile(calc_layer_weight_gradient_kernel);
    calc_input_weight_gradient_shader = lc::device.compile(calc_input_weight_gradient_kernel);
    calc_output_weight_gradient_shader = lc::device.compile(calc_output_weight_gradient_kernel);

    splitk_reduce_shader = lc::device.compile(splitk_reduce_kernel);

    apply_ema_weight_shader = lc::device.compile(apply_ema_weight_kernel);
}


void init_weight() {
    lc::stream << init_weight_shader(weight_buffer).dispatch(weight_buffer.size());
}

void input_forward(uint batch_size, BufferView<half4> &weight, Buffer<half4> &input, Buffer<half4> &store) {
    lc::stream << input_forward_shader(batch_size, weight, input, store).dispatch(batch_size / 2);
}
void layer_forward(uint batch_size, BufferView<half4> &weight, Buffer<half4> &layer_act, Buffer<half4> &store) {
    lc::stream << layer_forward_shader(batch_size, weight, layer_act, store).dispatch(batch_size / 2);
}
void output_forward(uint batch_size, BufferView<half4> &weight, Buffer<half4> &layer_act, Buffer<half4> &store) {
    lc::stream << output_forward_shader(batch_size, weight, layer_act, store).dispatch(batch_size / 4);
}

void layer_backward(uint batch_size, uint layer_id, Buffer<half4> &layer_grad, Buffer<half4> &fwd_act, Buffer<half4> &store) {
    lc::stream << layer_backward_shader(batch_size, layer_weight[layer_id], layer_grad, fwd_act, store).dispatch(batch_size / 2);
}
void output_backward(uint batch_size, Buffer<half4> &output_grad, Buffer<half4> &fwd_act, Buffer<half4> &store) {
    lc::stream << output_backward_shader(batch_size, output_weight, output_grad, fwd_act, store).dispatch(batch_size / 2);
}

void encode_input_backward(uint batch_size, Buffer<half4> &bwd_tmp0, Buffer<half4> &input_grad) {
    lc::stream << encode_input_backward_shader(batch_size, input_weight, bwd_tmp0, input_grad).dispatch(batch_size / 2);
}

void calc_output_gradiant(uint batch_size, Buffer<half4> &output, Buffer<half4> &target, Buffer<half4> &output_grad, Buffer<half4> &output_loss) {
    lc::stream << calc_output_gradient_shader(batch_size, output, target, output_grad, output_loss).dispatch(batch_size / 4);
}

void calc_layer_weight_gradient(uint batch_size, BufferView<half4> &weight_grad, Buffer<half4> &fwd_act, Buffer<half4> &layer_grad, Buffer<half4> &splitk_tmp) {
    uint dsp_size = layer_width*layer_width/4;
    lc::stream << calc_layer_weight_gradient_shader(batch_size, layer_grad, fwd_act, splitk_tmp).dispatch(batch_size / 4)
        << splitk_reduce_shader(batch_size, dsp_size, splitk_tmp, weight_grad).dispatch(dsp_size);
}
void calc_input_weight_gradient(uint batch_size, Buffer<half4> &input, Buffer<half4> &layer_grad, Buffer<half4> &splitk_tmp) {
    lc::stream << calc_input_weight_gradient_shader(batch_size, layer_grad, input, splitk_tmp).dispatch(32*batch_size/tile_k)
        << splitk_reduce_shader(batch_size, 16, splitk_tmp, input_weight_gradient).dispatch(16);
}
void calc_output_weight_gradient(uint batch_size, Buffer<half4> &fwd_act, Buffer<half4> &ouput_grad, Buffer<half4> &splitk_tmp) {
    lc::stream << calc_output_weight_gradient_shader(batch_size, fwd_act, ouput_grad, splitk_tmp).dispatch(32*batch_size/tile_k)
        << splitk_reduce_shader(batch_size, 24, splitk_tmp, output_weight_gradient).dispatch(24);
}

void apply_ema_weight() {
    lc::stream << apply_ema_weight_shader(weight_buffer, ema_weight_buffer, step).dispatch(weight_buffer.size());
}

}

namespace trainer {

Kernel1D prepare_train_data_kernel = []($bindless heap, $buffer<half4> input, $buffer<half4> target, $uint ofs) {
    set_block_size(256);
    $uint tid = $dispatch_x;
    $float2 t1 = sobol_2d(233 + tid*2 + ofs*train_batch_size);
    $float2 t2 = sobol_2d(233 + tid*2 + 1 + ofs*train_batch_size);
    $half4 s1 = heap.tex2d(0).sample(t1);
    $half4 s2 = heap.tex2d(0).sample(t2);
    $half4 in = make_float4(t1.x, t1.y, t2.x, t2.y);
    input.write(tid, in);
    target.write(tid*2, s1);
    target.write(tid*2 + 1, s2);
};

Kernel1D init_inference_input_kernel = []($buffer<half4> inference_input) {
    $uint tid = $dispatch_x;
    $half4 tmp;
    for (int t = 0; t < 2; t++) {
        $uint idx = tid*2 + t;
        $uint x = idx % 1920;
        $uint y = idx / 1920;
        tmp[t*2] = (x + 0.5) / 1920;
        tmp[t*2 + 1] = (y + 0.5) / 1080;
    }
    inference_input.write(tid, tmp);
};

Kernel2D fetch_inference_output_kernel = []($buffer<half4> inference_output, $image<float> inference_image) {
    $uint x = $dispatch_x;
    $uint y = $dispatch_y;
    $float4 c = inference_output.read(x + y*1920);
    inference_image.write($uint2{x, y}, c);
};

Shader1D<BindlessArray, Buffer<half4>, Buffer<half4>, uint> prepare_train_data_shader;
Shader1D<Buffer<half4>> init_inference_input_shader;
Shader2D<Buffer<half4>, Image<float>> fetch_inference_output_shader;

void init_buffer() {
    for (int i = 0; i < network::hidden_layers; i++) {
        fwd_tmp[i] = lc::device.create_buffer<half4>(network::layer_width * train_batch_size / 4);
    }
#if use_input_encode
    fwd_tmp[network::hidden_layers] = lc::device.create_buffer<half4>(encoder::encode_width * train_batch_size / 4);
#endif
    for (int i = 0; i < network::hidden_layers; i++) {
        bwd_tmp[i] = lc::device.create_buffer<half4>(network::layer_width * train_batch_size / 4);
    }
#if use_instant_ngp
    bwd_tmp[network::hidden_layers] = lc::device.create_buffer<half4>(encoder::encode_width * train_batch_size / 4);
#endif
    input = lc::device.create_buffer<half4>(input_dim * train_batch_size / 4);
    output = lc::device.create_buffer<half4>(output_dim_pad * train_batch_size / 4);
    target = lc::device.create_buffer<half4>(output_dim_pad * train_batch_size / 4);
    output_grad = lc::device.create_buffer<half4>(output_dim * train_batch_size / 4);
    output_loss = lc::device.create_buffer<half4>(output_dim * train_batch_size / 4);

    for (int i = 0; i < network::hidden_layers - 1; i++) {
        splitk_tmp[i] = lc::device.create_buffer<half4>(network::layer_width * network::layer_width * (train_batch_size / network::tile_k));
    }
#if use_input_encode
    splitk_tmp[network::hidden_layers - 1] = lc::device.create_buffer<half4>(network::layer_width * encoder::encode_width * (train_batch_size / network::tile_k));
#else
    splitk_tmp[network::hidden_layers - 1] = lc::device.create_buffer<half4>(network::layer_width * input_dim * (train_batch_size / network::tile_k));
#endif
    splitk_tmp[network::hidden_layers] = lc::device.create_buffer<half4>(network::layer_width * output_dim * (train_batch_size / network::tile_k));

    inference_image = lc::device.create_image<float>(PixelStorage::BYTE4, 1920, 1080);
    inference_input = lc::device.create_buffer<half4>(1920*1080*input_dim/4);
    inference_output = lc::device.create_buffer<half4>(1920*1080*output_dim_pad/4);
    inference_tmp = lc::device.create_buffer<half4>(1920*1080*network::layer_width/4);

    lc::stream << init_inference_input_shader(inference_input).dispatch(inference_input.size());
}

void init_shader() {
    prepare_train_data_shader = lc::device.compile(prepare_train_data_kernel);
    init_inference_input_shader = lc::device.compile(init_inference_input_kernel);
    fetch_inference_output_shader = lc::device.compile(fetch_inference_output_kernel);
}

void prepare_train_data() {
    static uint t = 0;
    lc::stream << prepare_train_data_shader(heap, input, target, t).dispatch(train_batch_size / 2);
    t++;
}

void inference() {
    network::inference(1920*1080, inference_input, inference_output, inference_tmp);
    lc::stream << fetch_inference_output_shader(inference_output, inference_image).dispatch(1920, 1080);
}

}

namespace optimizer {

Kernel1D init_buffer_kernel = []($buffer<half4> weights, $buffer<float4> weights_fp, $buffer<float4> mt, $buffer<float4> vt) {
    $uint tid = $dispatch_x;
    $half4 w = weights.read(tid);
    $float4 w_fp = w;
    weights_fp.write(tid, w_fp);
    mt.write(tid, make_float4(0));
    vt.write(tid, make_float4(0));
};
Shader1D<Buffer<half4>, Buffer<float4>, Buffer<float4>, Buffer<float4>> init_buffer_shader;

void init_shader() {
    init_buffer_shader = lc::device.compile(init_buffer_kernel);
    optimize_shader = lc::device.compile(optimize_kernel);
}

void init_buffer(Buffer<half4> &weights) {
    uint n_param = weights.size() * 4;
    weights_fp = lc::device.create_buffer<float4>(n_param / 4);
    mt = lc::device.create_buffer<float4>(n_param / 4);
    vt = lc::device.create_buffer<float4>(n_param / 4);

    lc::stream << init_buffer_shader(weights, weights_fp, mt, vt).dispatch(n_param / 4);
}

}