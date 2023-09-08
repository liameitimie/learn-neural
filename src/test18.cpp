#include <luisa/luisa-compute.h>
#include <luisa/gui/window.h>
// #include <luisa/vstl/common.h>
// #include <luisa/core/logging.h>
// #include <luisa/runtime/context.h>
// #include <luisa/runtime/device.h>
// #include <luisa/runtime/stream.h>
#include <stb/stb_image.h>
#include <iostream>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

#define use_output_bias 0

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

float as_uniform(uint x) {
    union {
        uint u;
        float f;
    } t;
    t.u = (x >> 9) | 0x3f800000u;
    return t.f - 1;
}

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
        return as_uniform(next_uint());
    }
}

uint tea(uint v0, uint v1) {
    uint s0 = 0;
    for (int i = 0; i < 8; i++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
};

float2 sobol_2d(uint i) {
    auto m0 = [](uint x) { return 1u << (31 - x); };
    auto m1 = [](uint x) {
        uint m = 1u << 31;
        for(int i = 0; i < 5; i++) {
            m |= (x >> i & 1) * (m >> (1 << i));
        }
        return m;
    };
    uint v0 = 0;
    uint v1 = 0;
    i ^= i >> 1;
    for(int j = 0; j < 32; j++) {
        if((i >> j & 1) != 0) {
            v0 ^= m0(j);
            v1 ^= m1(j);
        }
    }
    return float2{as_uniform(v0), as_uniform(v1)};
};


struct img {
    int width = 0;
    int height = 0;
    int channel = 0;
    stbi_uc* data;

    void load(const char* file) {
        auto pixel = stbi_load(file, &width, &height, &channel, 4);
        data = pixel;
    }
    float4 read(int x, int y) {
        if (x < 0 || x >= width || y < 0 || y > width) return float4{0};
        stbi_uc* p = data + (x + y*width)*4;
        return float4{p[0]/255.0f, p[1]/255.0f, p[2]/255.0f, p[3]/255.0f};
    }
    float4 read(float x, float y) {
        int tx = x * width;
        int ty = y * height;
        return read(tx, ty);
    }
    float4 read(float2 s) {
        return read(s.x, s.y);
    }
};

enum Layout{
    ColMajor = 0,
    RowMajor = 1
};

struct mat {
    int rows = 0;
    int cols = 0;
    int size = 0;
    Layout layout = ColMajor;
    float* data = nullptr;

    mat() {}
    mat(int r, int c, Layout l, float* p = nullptr) {
        rows = r, cols = c, size = r * c;
        layout = l;
        if (p) data = p;
        else data = new float [r * c];
    }
    float& operator()(int r, int c) {
        assert(r >=0 && r < rows);
        assert(c >=0 && c < cols);
        if (r < 0 || r >= rows || c < 0 || c > cols) {
            fmt::print("error: read [{}, {}] in [{}, {}] mat\n", r, c, rows, cols);
            exit(1);
        }
        switch (layout) {
            case ColMajor: return data[r + c * rows];
            case RowMajor: return data[c + r * cols];
        }
    }
    void transpose() {
        std::swap(rows, cols);
        layout = (Layout)(layout ^ 1);
    }
    void print() {
        fmt::print("rows: {}, cols: {}\n", rows, cols);
        for (int i = 0; i < rows; i++) {
            fmt::print("r{}: [", i);
            for (int j = 0; j < cols; j++) {
                fmt::print("{}", operator()(i, j));
                if (j < cols - 1) fmt::print(", ");
            }
            fmt::print("]\n");
        }
    }
};

void matmul(mat &a, mat &b, mat &c) {
    assert(a.cols == b.rows);
    assert(a.rows == c.rows);
    assert(b.cols == c.cols);
    if(a.cols != b.rows || a.rows != c.rows || b.cols != c.cols) {
        print("error: matmul [{}, {}] x [{}, {}] -> [{}, {}]", a.rows, a.cols, b.rows, b.cols, c.rows, c.cols);
        exit(1);
    }
    for (int i = 0; i < c.rows * c.cols; i++) c.data[i] = 0;
    for (int x = 0; x < c.rows; x++) {
        for (int k = 0; k < a.cols; k++) {
            float t = a(x, k);
            for (int y = 0; y < c.cols; y++) {
                c(x, y) += t * b(k, y);
            }
        }
    }
}

const uint input_dim = 2;
const uint output_dim = 1;
const uint output_dim_pad = 4;

enum class Activation {
	ReLU,
	LeakyReLU,
	Sine,
	None,
};
const Activation activation = Activation::ReLU;

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
        case Activation::Sine: x *= cos(asin(fwd_act));break;
        case Activation::None:break;
    }
}

enum class Loss {
    L2,
    RelativeL2
};
const Loss loss = Loss::L2;

float prediction_loss_host(float prediction, float target, int batch_size) {
    switch (loss) {
        case Loss::L2: return (prediction - target) * (prediction - target) / batch_size;
        case Loss::RelativeL2: return (prediction - target) * (prediction - target) / (prediction * prediction + 0.01) / batch_size;
    }
}

float loss_gradient_host(float prediction, float target, int batch_size) {
    switch (loss) {
        case Loss::L2: return 2 * (prediction - target) / batch_size;
        case Loss::RelativeL2: return 2 * (prediction - target) / (prediction * prediction + 0.01) / batch_size;
    }
}

namespace optimizer{
    int n_param = 0;
    float learning_rate = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;
    uint t = 0;

    float *mt;
    float *vt;

    void init(int _n_param) {
        n_param = _n_param;
        mt = new float [n_param];
        vt = new float [n_param];
        for (int i = 0; i < n_param; i++) {
            mt[i] = 0;
            vt[i] = 0;
        }
    }
    void optimize_adam(int i, float* weights, float* gradient) {
        float w = weights[i];
        float g = gradient[i];

        float m = mt[i] = beta1*mt[i] + (1 - beta1)*g;
        float v = vt[i] = beta2*vt[i] + (1 - beta2)*g*g;

        m = m / (1 - pow(beta1, t));
        v = v / (1 - pow(beta2, t));
        w = w - learning_rate * m / (sqrt(v) + eps);

        weights[i] = w;
    }
    void optimize_sgd(int i, float* weights, float* gradient) {
        weights[i] -= learning_rate * gradient[i];
    }

    void optimize(float* weights, float* gradient) {
        t++;
        for (int i = 0; i < n_param; i++) {
            optimize_adam(i, weights, gradient);
            // optimize_sgd(i, weights, gradient);
        }
    }
}

namespace network {
    const uint layer_width = 32;
    const uint hidden_layers = 5;

    float* weight_buffer;
    mat layer_weight[hidden_layers - 1];
    mat input_weight;
    mat output_weight;
    mat layer_bias[hidden_layers];

    

    float* gradient_buffer;
    mat layer_weight_gradient[hidden_layers - 1];
    mat input_weight_gradient;
    mat output_weight_gradient;
    mat layer_bias_gradient[hidden_layers];
    
#if use_output_bias
    mat output_bias;
    mat output_bias_gradient;
#endif

    void init_buffer();
    void init_weight();
    void init();

    void print_weight();
    void print_gradient();
    void print(string name, mat& m) {
        fmt::print("{}: ", name);
        m.print();
    }

    void apply_forward_activation(mat &v) {
        // fmt::print("apply_forward_activation:\n");
        // print("pre act v", v);

        for (int i = 0; i < v.rows*v.cols; i++) {
            activation_forward_host(v.data[i]);
        }

        // print("post act v", v);
        // fmt::print("\n");
    }
    void apply_backward_activation(mat &g, mat &act) {
        assert(g.rows == act.rows && g.cols == act.cols && g.layout == act.layout);
        if (g.rows != act.rows || g.cols != act.cols || g.layout != act.layout) {
            fmt::print("error apply_backward_activation\n");
            exit(0);
        }
        // fmt::print("apply_backward_activation:\n");
        // print("pre act gradient", g);
        // print("fwd act", act);

        for (int i = 0; i < g.rows*g.cols; i++) {
            activation_backward_host(g.data[i], act.data[i]);
        }
        // print("post act gradient", g);
        // fmt::print("\n");
    }

    void calc_output_gradiant(uint batch_size, mat &output, mat &target, mat &output_grad, mat &output_loss) {
        // fmt::print("calc_output_gradiant:\n");
        // print("output", output);
        // print("target", target);
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < output_dim; j++) {
                output_loss(j, i) = prediction_loss_host(output(j, i), target(j, i), batch_size);
                output_grad(j, i) = loss_gradient_host(output(j, i), target(j, i), batch_size);
            }
        }
        // print("output_grad", output_grad);
        // print("output_loss", output_loss);
        // fmt::print("\n");
    }

    void apply_layer_bias(uint batch_size, mat &fwd_tmp, int layer_id) {
        // fmt::print("apply_layer_bias {}\n", layer_id);
        // print("fwd_tmp", fwd_tmp);
        // print("layer_bias[layer_id]", layer_bias[layer_id]);

        for (int i = 0; i < layer_width; i++) {
            for (int j = 0; j < batch_size; j++) {
                fwd_tmp(i, j) += layer_bias[layer_id](i, 0);
            }
        }
        // print("fwd_tmp_biased", fwd_tmp);
        // fmt::print("\n");
    }
    void calc_layer_bias_gradient(uint batch_size, mat &bwd_tmp, int layer_id) {
        // fmt::print("calc_layer_bias_gradient {}\n", layer_id);
        for (int i = 0; i < layer_bias_gradient[layer_id].size; i++) {
            layer_bias_gradient[layer_id].data[i] = 0;
        }
        // print("bwd_tmp", bwd_tmp);
        for (int i = 0; i < layer_width; i++) {
            for (int j = 0; j < batch_size; j++) {
                layer_bias_gradient[layer_id](i, 0) += bwd_tmp(i, j);
            }
        }
        // print("layer_bias_gradient[layer_id]", layer_bias_gradient[layer_id]);
        // fmt::print("\n");
    }


    void apply_output_bias(uint batch_size, mat& output) {
#if use_output_bias
        // fmt::print("apply_output_bias\n");
        // print("output", output);
        // print("output_bias", output_bias);

        for (int j = 0; j < batch_size; j++) {
            for (int i = 0; i < output_dim; i++) {
                output(i, j) += output_bias(i, 0);
            }
        }
        // print("output_biased", output);
        // fmt::print("\n");
#endif
    }
    
    void calc_output_bias_gradient(uint batch_size, mat &output_grad) {
#if use_output_bias
        // fmt::print("calc_output_bias_gradient\n");
        for (int i = 0; i < output_bias_gradient.size; i++) {
            output_bias_gradient.data[i] = 0;
        }
        // print("output_grad", output_grad);
        for (int i = 0; i < output_dim; i++) {
            for (int j = 0; j < batch_size; j++) {
                output_bias_gradient(i, 0) += output_grad(i, j);
            }
        }
        // print("output_bias_gradient", output_bias_gradient);
        // fmt::print("\n");
#endif
    }


    void train_step(uint batch_size, mat &input, mat &target, mat &output, mat &output_grad, mat &output_loss, mat *fwd_tmp, mat *bwd_tmp) {
        const uint n = hidden_layers;
        // fmt::print("\n");
        // forward
        // fmt::print("input forward matmul\n");
        // print("input_weight", input_weight);
        // print("input", input);
        matmul(input_weight, input, fwd_tmp[0]);
        // print("fwd_tmp[0]", fwd_tmp[0]);
        // fmt::print("\n");

        apply_layer_bias(batch_size, fwd_tmp[0], 0);
        apply_forward_activation(fwd_tmp[0]);

        for (int i = 0; i < n - 1; i++) {

            // fmt::print("layer {} forward matmul\n", i);
            // print("layer_weight[i]", layer_weight[i]);
            // print("fwd_tmp[i]", fwd_tmp[i]);
            matmul(layer_weight[i], fwd_tmp[i], fwd_tmp[i + 1]);
            // print("fwd_tmp[i + 1]", fwd_tmp[i + 1]);
            // fmt::print("\n");

            apply_layer_bias(batch_size, fwd_tmp[i + 1], i + 1);
            apply_forward_activation(fwd_tmp[i + 1]);
        }
        // fmt::print("output forward matmul\n");
        // print("output_weight", output_weight);
        // print("fwd_tmp[n - 1]", fwd_tmp[n - 1]);
        matmul(output_weight, fwd_tmp[n - 1], output);
        // print("output", output);
        // fmt::print("\n");

        apply_output_bias(batch_size, output);

        calc_output_gradiant(batch_size, output, target, output_grad, output_loss);

        // backward
        for (int i = 0; i < n - 1; i++) layer_weight[i].transpose();
        input_weight.transpose();
        output_weight.transpose();

        // fmt::print("output backward matmul\n");
        // print("output_weight", output_weight);
        // print("output_grad", output_grad);
        matmul(output_weight, output_grad, bwd_tmp[n - 1]);
        // print("bwd_tmp[n - 1]", bwd_tmp[n - 1]);
        // fmt::print("\n");

        apply_backward_activation(bwd_tmp[n - 1], fwd_tmp[n - 1]);
        for (int i = n - 1; i > 0; i--) {
            // fmt::print("layer {} backward matmul\n", i);
            // print("layer_weight[i - 1]", layer_weight[i - 1]);
            // print("bwd_tmp[i]", bwd_tmp[i]);
            matmul(layer_weight[i - 1], bwd_tmp[i], bwd_tmp[i - 1]);
            // print("bwd_tmp[i - 1]", bwd_tmp[i - 1]);
            // fmt::print("\n");

            apply_backward_activation(bwd_tmp[i - 1], fwd_tmp[i - 1]);
        }

        for (int i = 0; i < n - 1; i++) layer_weight[i].transpose();
        input_weight.transpose();
        output_weight.transpose();

        // calc weight gradient
        input.transpose();
        for (int i = 0; i < n; i++) fwd_tmp[i].transpose();

        // fmt::print("calc input weight gradient matmul\n");
        // print("bwd_tmp[0]", bwd_tmp[0]);
        // print("input", input);
        matmul(bwd_tmp[0], input, input_weight_gradient);
        // print("input_weight_gradient", input_weight_gradient);
        // fmt::print("\n");

        for (int i = 0; i < n - 1; i++) {
            // fmt::print("calc layer {} weight gradient matmul\n", i);
            // print("bwd_tmp[i + 1]", bwd_tmp[i + 1]);
            // print("fwd_tmp[i]", fwd_tmp[i]);
            matmul(bwd_tmp[i + 1], fwd_tmp[i], layer_weight_gradient[i]);
            // print("layer_weight_gradient[i]", layer_weight_gradient[i]);
            // fmt::print("\n");
        }
        // fmt::print("calc output weight gradient matmul\n");
        // print("output_grad", output_grad);
        // print("fwd_tmp[n - 1]", fwd_tmp[n - 1]);
        matmul(output_grad, fwd_tmp[n - 1], output_weight_gradient);
        // print("output_weight_gradient", output_weight_gradient);
        // fmt::print("\n");

        input.transpose();
        for (int i = 0; i < n; i++) fwd_tmp[i].transpose();

        calc_output_bias_gradient(batch_size, output_grad);
        for (int i = 0; i < hidden_layers; i++) {
            calc_layer_bias_gradient(batch_size, bwd_tmp[i], i);
        }

        // fmt::print("optimize\n");
        // print_weight();
        // print_gradient();
        optimizer::optimize(weight_buffer, gradient_buffer);
        // print_weight();
    }

    void inference(uint batch_size, mat &input, mat &output, mat *fwd_tmp) {
        const uint n = hidden_layers;
        matmul(input_weight, input, fwd_tmp[0]);
        apply_layer_bias(batch_size, fwd_tmp[0], 0);
        apply_forward_activation(fwd_tmp[0]);
        for (int i = 0; i < n - 1; i++) {
            matmul(layer_weight[i], fwd_tmp[i & 1], fwd_tmp[~i & 1]);
            apply_layer_bias(batch_size, fwd_tmp[~i & 1], i + 1);
            apply_forward_activation(fwd_tmp[~i & 1]);
        }
        matmul(output_weight, fwd_tmp[(n - 1)&1], output);
        apply_output_bias(batch_size, output);
        
        // matmul(input_weight, input, fwd_tmp[0]);
        // apply_bias(batch_size, fwd_tmp[0], 0);
        // apply_forward_activation(fwd_tmp[0]);
        // for (int i = 0; i < n - 1; i++) {
        //     matmul(layer_weight[i], fwd_tmp[i & 1], fwd_tmp[~i & 1]);
        //     apply_bias(batch_size, fwd_tmp[~i & 1], i + 1);
        //     apply_forward_activation(fwd_tmp[~i & 1]);
        // }
        // matmul(output_weight, fwd_tmp[(n - 1)&1], output);
    }

    
}

namespace trainer {
    uint train_batch_size = 16384;
    mat fwd_tmp[network::hidden_layers];
    mat bwd_tmp[network::hidden_layers];
    mat output;
    mat output_grad;
    mat output_loss;
    mat input;
    mat target;

    img train_image;

    const uint2 inference_res = uint2{200, 200};
    const uint inference_size = inference_res.x * inference_res.y;

    Image<float> gpu_inference_image;
    stbi_uc* inference_image;
    mat inference_input;
    mat inference_output;
    mat inference_tmp[2];

    void init_buffer();
    void init(const char* file, uint batch_size = 16384) {
        train_batch_size = batch_size;
        train_image.load(file);
        init_buffer();
        network::init();
    }

    void prepare_train_data() {
        static uint t = 0;
        for (int i = 0; i < train_batch_size; i++) {
            float2 s = sobol_2d(233 + i + t*train_batch_size);
            float4 c = train_image.read(s);
            for (int j = 0; j < 2; j++) input(j, i) = s[j] + 1;

            if (activation == Activation::Sine) {
                for (int j = 0; j < 3; j++) target(j, i) = c[j] - 0.5;
            }
            else {
                for (int j = 0; j < 3; j++) target(j, i) = c[j];
            }
        }
        t++;

        // print("train data:\n");
        // for (int i = 0; i < 32; i++) {
        //     print("[{}, {}] -> [{}, {}, {}]\n", input(0, i), input(1, i), target(0, i), target(1, i), target(2, i));
        // }
        // print("\n");
    }

    void train_step() {
        prepare_train_data();
        network::train_step(train_batch_size, input, target, output, output_grad, output_loss, fwd_tmp, bwd_tmp);

        float loss = 0;
        for (int i = 0; i < train_batch_size; i++) {
            for (int j = 0; j < output_dim; j++) {
                loss += output_loss(j, i);
            }
        }
        print("loss: {}\n", loss);
    }

    void inference() {
        network::inference(inference_size, inference_input, inference_output, inference_tmp);
        for (int i = 0; i < inference_size; i++) {
            if (output_dim == 1) {
                float3 c1 = float3{255, 165, 0};
                float3 c2 = float3{65, 105, 225};
                float3 c3 = float3{255, 255, 255};
                float x = inference_output(0, i);
                float3 c;
                if (x < 0) c = lerp(c3, c1, float3(-x));
                else c = lerp(c3, c2, float3(x));

                for (int j = 0; j < 3; j++) {
                    inference_image[j + i*4] = clamp((int)round(c[j]), 0, 255);
                }
            }
            else {
                for (int j = 0; j < output_dim; j++) {
                    inference_image[j + i*4] = clamp((int)round(255 * (inference_output(j, i) + 0.5*(activation == Activation::Sine))), 0, 255);
                }
            }
        }
        lc::stream << gpu_inference_image.copy_from(inference_image);
    }
}

// namespace gpu_network {
//     const uint2 inference_size = uint2{1920, 1080};
//     // for gpu inference
//     Buffer<half4> weight_buffer;
//     BufferView<half4> layer_weight[network::hidden_layers - 1];
//     BufferView<half4> input_weight;
//     BufferView<half4> output_weight;

//     Image<float> inference_image;
//     Buffer<half4> inference_input;
//     Buffer<half4> inference_output;
//     Buffer<half4> inference_tmp;

//     void init_buffer();
//     void init_shader();
//     void init();
//     void refresh_weights();
//     void inference();
// }

int main(int argc, char *argv[]) {
    lc::init(argv[0]);

    const int custom_samples = 32;
    trainer::init("assets/nahida.jpeg", custom_samples);

    // float w_in[4][2] = {
    //     {-0.65, 0.33},
    //     {0.093, 0.74},
    //     {0.16, 0.41},
    //     {0.65, 0.33}
    // };
    // float b_h0[4] = {
    //     1.4, 1.3, -0.0003, 1.2
    // };
    // float w_l0[4][4] = {
    //     {0.57, 1.1, -0.3, 0.38},
    //     {0.52, 0.67, -0.61, 0.16},
    //     {-0.3, 0.094, -0.43, -0.39},
    //     {-0.86, -0.4, 0.1, -0.76}
    // };
    // float b_h1[4] = {
    //     -0.073, -0.079, 0.14, 0.20
    // };
    // float w_out[4] = {
    //     1.2, 0.93, -0.25, -1.1
    // };
    // float b_out[1] = {
    //     -0.4
    // };
    // float w_in[4][2] = {
    //     {-0.25, -0.064},
    //     {-0.099, -0.33},
    //     {0.047, -0.26},
    //     {0.40, -0.3}
    // };
    // float b_h0[4] = {
    //     0.1, 0.1, 0.1, 0.1
    // };
    // float w_l0[4][4] = {
    //     {-0.067, -0.008, -0.2, 0.47},
    //     {-0.19, 0.47, -0.22, -0.31},
    //     {-0.36, 0.04, -0.34, -0.26},
    //     {-0.1, -0.17, -0.17, -0.3}
    // };
    // float b_h1[4] = {
    //     0.1, 0.1, 0.1, 0.1
    // };
    // float w_out[4] = {
    //     0.36, 0.49, -0.5, -0.11
    // };
    // float b_out[1] = {
    //     0.1
    // };

    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 2; j++) {
    //         network::input_weight(i, j) = w_in[i][j];
    //     }
    // }
    // for (int i = 0; i < 4; i++) {
    //     network::layer_bias[0](i, 0) = b_h0[i];
    // }
    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 4; j++) {
    //         network::layer_weight[0](i, j) = w_l0[i][j];
    //     }
    // }
    // for (int i = 0; i < 4; i++) {
    //     network::layer_bias[1](i, 0) = b_h1[i];
    // }
    // for (int i = 0; i < 4; i++) {
    //     network::output_weight(0, i) = w_out[i];
    // }
    // for (int i = 0; i < output_dim; i++) {
    //     network::output_bias(i, 0) = b_out[i];
    // }

    
    // network::print_weight();
    // trainer::inference();

    // gpu_network::init();

    // for (int i = 0; i < 1000000; i++) {
    //     trainer::train_step();
    // }

    // trainer::prepare_train_data();
    for (int i = 0; i < custom_samples; i++) {
        float2 s = sobol_2d(i + 233);
        float t = s.x * 2 * 3.1415926;
        float r = s.y;
        if (r < 0.5) {
            r = r*0.6;
        }
        else {
            r = (r-0.5)*0.6 + 0.5;
        }
        
        float2 ts;
        ts.x = 6 * r*cos(t), ts.y = 6 * r*sin(t);
        
        float c = s.y < 0.5 ? 1 : -1;

        print("[{}, {}] -> [{}]\n", ts.x, ts.y, c);
        
        // float3 c1 = float3(0.2, 0.5, 1.0);
        // float3 c2 = float3(1.0, 0.5, 0.2);
        // float3 c = lerp(c1, c2, float3(1 / (1 + exp(5 - 10*s.y))));
        // float3 c = r < 0.5 ? c1 : c2;

        for (int j = 0; j < 2; j++) trainer::input(j, i) = ts[j];
        trainer::target(0, i) = c;

        float3 c1 = float3{255, 165, 0};
        float3 c2 = float3{65, 105, 225};
        float3 c3 = float3{255, 255, 255};
        float3 tc;
        if (c < 0) tc = lerp(c3, c1, float3(-c));
        else tc = lerp(c3, c2, float3(c));

        // int x = (ts.x/12 + 0.5) * trainer::inference_res.x;
        // int y = (ts.y/12 + 0.5) * trainer::inference_res.y;
        // // print("x: {}, y: {}\n", x, y);
        // for (int j = 0; j < 3; j++) {
        //     trainer::inference_image[j + (x + y*trainer::inference_res.x)*4] = clamp((int)round(tc[j]), 0, 255);
        // }
        // lc::stream << trainer::gpu_inference_image.copy_from(trainer::inference_image);

        // if (activation == Activation::Sine) {
        //     for (int j = 0; j < 3; j++) trainer::target(j, i) = c[j] - 0.5;
        // }
        // else {
        //     for (int j = 0; j < 3; j++) trainer::target(j, i) = c[j];
        // }
    }
    // print("1\n");

    // for (int i = 0; i < 2; i++) {
    //     network::train_step(trainer::train_batch_size, trainer::input, trainer::target, trainer::output, trainer::output_grad, trainer::output_loss, trainer::fwd_tmp, trainer::bwd_tmp);
    // }
    
    Window window{"", trainer::inference_res};
    Swapchain swapchain = lc::device.create_swapchain(
        window.native_handle(),
        lc::stream,
        trainer::inference_res,
        false, false,
        3
    );

    int t = 0;
    while(!window.should_close()){
        // trainer::train_step();
        network::train_step(trainer::train_batch_size, trainer::input, trainer::target, trainer::output, trainer::output_grad, trainer::output_loss, trainer::fwd_tmp, trainer::bwd_tmp);

        float loss = 0;
        for (int i = 0; i < trainer::train_batch_size; i++) {
            for (int j = 0; j < output_dim; j++) {
                loss += trainer::output_loss(j, i);
            }
        }
        print("step {}, loss: {}\n", t, loss);
        t++;

        trainer::inference();
        lc::stream << swapchain.present(trainer::gpu_inference_image);
        // gpu_network::inference();
        // lc::stream << swapchain.present(gpu_network::inference_image);
        window.poll_events();
    }
    lc::stream.synchronize();

    // network::print_weight();
    // print("\n");
    // network::print_gradient();

    return 0;
}

namespace network {
    
    const uint layer_weight_size = layer_width * layer_width;
    const uint input_weight_size = input_dim * layer_width;
    const uint output_weight_size = output_dim * layer_width;

    const uint layer_bias_size = layer_width;
    const uint output_bias_size = output_dim;

    const uint weight_size = 
        layer_weight_size * (hidden_layers - 1)
        + input_weight_size
        + output_weight_size
        + layer_bias_size * hidden_layers
#if use_output_bias
        + output_bias_size
#endif
    ;

    void init_buffer() {
        weight_buffer = new float [weight_size];
        gradient_buffer = new float [weight_size];

        uint offset = 0;
        for (int i = 0; i < hidden_layers - 1; i++) {
            layer_weight[i] = mat(layer_width, layer_width, ColMajor, weight_buffer + offset);
            layer_weight_gradient[i] = mat(layer_width, layer_width, ColMajor, gradient_buffer + offset);
            offset += layer_weight_size;
        }
        
        input_weight = mat(layer_width, input_dim, ColMajor, weight_buffer + offset);
        input_weight_gradient = mat(layer_width, input_dim, ColMajor, gradient_buffer + offset);
        offset += input_weight_size;

        output_weight = mat(output_dim, layer_width, RowMajor, weight_buffer + offset);
        output_weight_gradient = mat(output_dim, layer_width, RowMajor, gradient_buffer + offset);
        offset += output_weight_size;

        for (int i = 0; i < hidden_layers; i++) {
            layer_bias[i] = mat(layer_width, 1, ColMajor, weight_buffer + offset);
            layer_bias_gradient[i] = mat(layer_width, 1, ColMajor, gradient_buffer + offset);
            offset += layer_bias_size;
        }
#if use_output_bias
        output_bias = mat(output_dim, 1, ColMajor, weight_buffer + offset);
        output_bias_gradient = mat(output_dim, 1, ColMajor, gradient_buffer + offset);
#endif
    }

    void init_weight() {
        float weight_scale[3];
        if (activation == Activation::None) {
            weight_scale[0] = sqrt(6.0 / layer_width);
            // weight_scale[1] = 30.0 / input_dim;
            weight_scale[1] = sqrt(6.0 / input_dim + layer_width);
            weight_scale[2] = sqrt(6.0 / layer_width);
        }
        else {
            weight_scale[0] = sqrt(6.0 / (layer_width + layer_width));
            weight_scale[1] = sqrt(6.0 / (input_dim + layer_width));
            weight_scale[2] = sqrt(6.0 / (layer_width + output_dim));
        }

        for (int t = 0; t < hidden_layers - 1; t++) {
            for (int i = 0; i < layer_weight[t].size; i++) {
                // layer_weight[t].data[i] = weight_scale[0] * (pcg32::next_float() * 2 - 1);
                layer_weight[t].data[i] = weight_scale[0] * (as_uniform(tea(t, i)) * 2 - 1);
            }
        }
        for (int i = 0; i < input_weight.size; i++) {
            input_weight.data[i] = weight_scale[1] * (as_uniform(tea(i, 233)) * 2 - 1);
        }
        for (int i = 0; i < output_weight.size; i++) {
            output_weight.data[i] = weight_scale[2] * (as_uniform(tea(i, 233)) * 2 - 1);
        }

        for (int t = 0; t < hidden_layers; t++) {
            for (int i = 0; i < layer_bias[t].size; i++) {
                layer_bias[t].data[i] = 0.1;
            }
        }
#if use_output_bias
        for (int i = 0; i < output_bias.size; i++) {
            output_bias.data[i] = 0.1;
        }
#endif
    }

    void init() {
        init_buffer();
        init_weight();
        optimizer::init(weight_size);
    }

    void print_weight() {
        fmt::print("input weight: ");
        input_weight.print();
        fmt::print("\n");
        fmt::print("layer bias {}: ", 0);
        layer_bias[0].print();
        fmt::print("\n");

        for (int i = 0; i < hidden_layers - 1; i++) {
            fmt::print("layer weight {}: ", i);
            layer_weight[i].print();
            fmt::print("\n");

            fmt::print("layer bias {}: ", i + 1);
            layer_bias[i + 1].print();
            fmt::print("\n");
        }
        fmt::print("output weight: ");
        output_weight.print();
        fmt::print("\n");
#if use_output_bias
        fmt::print("output bias: ");
        output_bias.print();
        fmt::print("\n");
#endif
    }

    void print_gradient() {
        fmt::print("input weight gradient: ");
        input_weight_gradient.print();
        fmt::print("\n");

        fmt::print("layer bias gradient {}: ", 0);
        layer_bias_gradient[0].print();
        fmt::print("\n");

        for (int i = 0; i < hidden_layers - 1; i++) {
            fmt::print("layer weight gradient {}: ", i);
            layer_weight_gradient[i].print();
            fmt::print("\n");

            fmt::print("layer bias gradient {}: ", i + 1);
            layer_bias_gradient[i + 1].print();
            fmt::print("\n");
        }

        fmt::print("output weight gradient: ");
        output_weight_gradient.print();
        fmt::print("\n");
#if use_output_bias
        fmt::print("output bias gradient: ");
        output_bias_gradient.print();
        fmt::print("\n");
#endif
    }
}

namespace trainer {

    void init_buffer() {
        for (auto& buf : fwd_tmp) {
            buf = mat(network::layer_width, train_batch_size, RowMajor);
        }
        for (auto& buf : bwd_tmp) {
            buf = mat(network::layer_width, train_batch_size, RowMajor);
        }
        input = mat(input_dim, train_batch_size, ColMajor);
        output = mat(output_dim, train_batch_size, ColMajor);
        target = mat(output_dim, train_batch_size, ColMajor);

        output_grad = mat(output_dim, train_batch_size, RowMajor);
        output_loss = mat(output_dim, train_batch_size, RowMajor);

        inference_image = new stbi_uc [inference_size * 4];
        memset(inference_image, 0, inference_size * 4);
        inference_input = mat(input_dim, inference_size, ColMajor);
        inference_output = mat(output_dim, inference_size, ColMajor);
        inference_tmp[0] = mat(network::layer_width, inference_size, RowMajor);
        inference_tmp[1] = mat(network::layer_width, inference_size, RowMajor);

        for (int i = 0; i < inference_size; i++) {
            int x = i % inference_res.x;
            int y = i / inference_res.x;
            inference_input(0, i) = -12 * (x + 0.5) / inference_res.x + 6;
            inference_input(1, i) = -12 * (y + 0.5) / inference_res.y + 6;
        }

        gpu_inference_image = lc::device.create_image<float>(PixelStorage::BYTE4, trainer::inference_res);
    }

}

// namespace gpu_network {
//     const uint layer_width = network::layer_width;
//     const uint inference_image_size = inference_size.x * inference_size.y;

//     void init_buffer() {
//         weight_buffer = lc::device.create_buffer<half4>(network::weight_size / 4);
//         uint offset = 0;
//         for (int i = 0; i < network::hidden_layers - 1; i++) {
//             layer_weight[i] = weight_buffer.view(offset, network::layer_weight_size / 4);
//             offset += network::layer_weight_size / 4;
//         }
//         input_weight = weight_buffer.view(offset, network::input_weight_size / 4);
//         offset += network::input_weight_size / 4;
//         output_weight = weight_buffer.view(offset, network::output_weight_size / 4);

//         inference_image = lc::device.create_image<float>(PixelStorage::BYTE4, inference_size);
//         inference_input = lc::device.create_buffer<half4>(inference_image_size*input_dim/4);
//         inference_output = lc::device.create_buffer<half4>(inference_image_size*output_dim_pad/4);
//         inference_tmp = lc::device.create_buffer<half4>(inference_image_size*network::layer_width/4);
//     }

//     void activation_forward($half &x) {
//         switch (activation) {
//             case Activation::ReLU: x *= (x > 0).cast<half>();break;
//             case Activation::LeakyReLU: x *= half(1) - half(0.9)*(x <= 0).cast<half>();break;
//             case Activation::Sine: x = sin(x);break;
//             case Activation::None:break;
//         }
//     }
//     void apply_activation_forward($half4 &acc) {
//         for (int i = 0; i < 4; i++) {
//             activation_forward(acc[i]);
//         }
//     }

//     void load_32x2($buffer<half4> &buf, $shared<half4> &smem) {
//         $uint tid = $dispatch_x % 256;
//         $if (tid < 16) {
//             smem[tid] = buf.read(tid);
//         };
//     }
//     void load_transpose_3x32($buffer<half4> &buf, $shared<half4> &smem) {
//         $uint tid = $dispatch_x % 256;
//         $if (tid < 24) {
//             $half4 tmp = buf.read(tid);
//             for (int i = 0; i < 4; i++) {
//                 smem[tid%8*4 + i][tid/8] = tmp[i];
//             }
//         };
//     }
//     void load_32x32($buffer<half4> &buf, $shared<half4> &smem) {
//         $uint tid = $dispatch_x % 256;
//         smem[tid] = buf.read(tid);
//     }
//     void load_transpose_512x2($buffer<half4> &buf, $shared<half4> &smem, $uint offset) {
//         $uint tid = $dispatch_x % 256;
//         $half4 tmp = buf.read(tid + offset);
//         for (int i = 0; i < 4; i++) {
//             smem[tid/2 + i%2*128][i/2 + tid%2*2] = tmp[i];
//         }
//     }
//     void load_512x4($buffer<half4> &buf, $shared<half4> &smem, $uint offset, $uint stride) {
//         $uint tid = $dispatch_x % 256;
//         smem[tid] = buf.read(offset + tid/128*stride);
//         smem[tid + 256] = buf.read(offset + (2 + tid/128)*stride);
//     }
//     void out_product_3x4_c($half4 &a, $half4 &b, $half4 *acc) {
//         for (int j = 0; j < 4; j++) {
//             for (int i = 0; i < 3; i++) {
//                 acc[j][i] += a[i] * b[j];
//             }
//         }
//     }
//     void out_product_8x8_r($half4 *a, $half4 *b, $half4 *acc) {
//         for (int tx = 0; tx < 2; tx++) {
//             for (int ty = 0; ty < 2; ty++) {
//                 for (int i = 0; i < 4; i++) {
//                     for (int j = 0; j < 4; j++) {
//                         acc[i + (ty + tx*2)*4][j] += a[tx][i] * b[ty][j];
//                     }
//                 }
//             }
//         }
//     }

//     Kernel1D output_forward_kernel = []($uint batch_size, $buffer<half4> weight, $buffer<half4> act, $buffer<half4> store) {
//         set_block_size(256);
//         $shared<half4> a_tile{32};
//         $half4 acc[4];
//         $half4 a_frag;
//         $half4 b_frag;

//         $uint tid = $dispatch_x % 256;
//         $uint bid = $dispatch_x / 256;

//         load_transpose_3x32(weight, a_tile);
//         sync_block();
//         $for (k, layer_width) {
//             a_frag = a_tile[k];
//             b_frag = act.read(tid + bid*256 + k*batch_size/4);
//             out_product_3x4_c(a_frag, b_frag, acc);
//         };
//         for (int i = 0; i < 4; i++) {
//             store.write(i + tid*4 + bid*256*4, acc[i]);
//         }
//     };

//     // 放弃挣扎
//     void store_res_tile($buffer<half4> &buf, $shared<half4> &smem, $half4 *acc, $uint stride, $uint tid, $uint bid) {
//         // $uint tid = $dispatch_x % 256;
//         // $uint bid = $dispatch_x / 256;
//         $uint x_ofs = tid%32/8;
//         $uint y_ofs = tid%8 + tid/32*16;
//         for (int t = 0; t < 2; t++) {
//             for (int i = 0; i < 4; i++) {
//                 sync_block();
//                 smem[y_ofs + x_ofs*128] = acc[i + t*2*4];
//                 smem[y_ofs + 8 + x_ofs*128] = acc[i + (t*2 + 1)*4];
//                 sync_block();

//                 buf.write(tid%128 + bid*128 + (i + t*16 + tid/128*4)*stride/4, smem[tid]);
//                 buf.write(tid%128 + bid*128 + (i + t*16 + (tid/128 + 2)*4)*stride/4, smem[tid + 256]);
//             }
//         }
//     }

//     Kernel1D layer_forward_kernel = []($uint batch_size, $buffer<half4> weight, $buffer<half4> act, $buffer<half4> store) {
//         set_block_size(256);
//         $shared<half4> a_tile{256};
//         $shared<half4> b_tile{512};
//         $half4 acc[16];
//         $half4 a_frag[2];
//         $half4 b_frag[2];

//         $uint tid = $dispatch_x % 256;
//         $uint bid = $dispatch_x / 256;

//         $uint x_ofs = tid%32/8;
//         $uint y_ofs = tid%8 + tid/32*16;

//         load_32x32(weight, a_tile);

//         $for (k, 0u, layer_width, 4u) {
//             sync_block();
//             load_512x4(act, b_tile, tid%128 + bid*128 + k*batch_size/4, batch_size/4);
//             sync_block();
//             $for (k1, 4) {
//                 for (int t = 0; t < 2; t++) {
//                     a_frag[t] = a_tile[x_ofs + t*4 + (k + k1)*8];
//                     b_frag[t] = b_tile[y_ofs + t*8 + k1*128];
//                 }
//                 out_product_8x8_r(a_frag, b_frag, acc);
//             };
//         };
//         for (int i = 0; i < 16; i++) {
//             apply_activation_forward(acc[i]);
//         }
//         store_res_tile(store, b_tile, acc, batch_size, tid, bid);
//     };

//     Kernel1D input_forward_kernel = []($uint batch_size, $buffer<half4> weight, $buffer<half4> input, $buffer<half4> store) {
//         set_block_size(256);
//         $shared<half4> a_tile{16};
//         $shared<half4> b_tile{512};
//         $half4 acc[16];
//         $half4 a_frag[2];
//         $half4 b_frag[2];

//         $uint tid = $dispatch_x % 256;
//         $uint bid = $dispatch_x / 256;

//         $uint x_ofs = tid%32/8;
//         $uint y_ofs = tid%8 + tid/32*16;

//         load_32x2(weight, a_tile);
//         load_transpose_512x2(input, b_tile, bid*256);
//         sync_block();

//         $for (k, input_dim) {
//             for (int t = 0; t < 2; t++) {
//                 a_frag[t] = a_tile[x_ofs + t*4 + k*8];
//                 b_frag[t] = b_tile[y_ofs + t*8 + k*128];
//             }
//             out_product_8x8_r(a_frag, b_frag, acc);
//         };
//         for (int i = 0; i < 16; i++) {
//             apply_activation_forward(acc[i]);
//         }
//         store_res_tile(store, b_tile, acc, batch_size, tid, bid);
//     };

//     Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>> output_forward_shader;
//     Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>> layer_forward_shader;
//     Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>> input_forward_shader;

//     Kernel1D init_inference_input_kernel = []($buffer<half4> inference_input) {
//         $uint tid = $dispatch_x;
//         $half4 tmp;
//         for (int t = 0; t < 2; t++) {
//             $uint idx = tid*2 + t;
//             $uint x = idx % 1920;
//             $uint y = idx / 1920;
//             tmp[t*2] = (x + 0.5) / 1920 + 1;
//             tmp[t*2 + 1] = (y + 0.5) / 1080 + 1;
//         }
//         inference_input.write(tid, tmp);
//     };
//     Kernel2D fetch_inference_output_kernel = []($buffer<half4> inference_output, $image<float> inference_image) {
//         $uint x = $dispatch_x;
//         $uint y = $dispatch_y;
//         $float4 c = inference_output.read(x + y*1920);
//         inference_image.write($uint2{x, y}, c);
//     };

//     Shader1D<Buffer<half4>> init_inference_input_shader;
//     Shader2D<Buffer<half4>, Image<float>> fetch_inference_output_shader;

//     void init_shader() {
//         output_forward_shader = lc::device.compile(output_forward_kernel);
//         layer_forward_shader = lc::device.compile(layer_forward_kernel);
//         input_forward_shader = lc::device.compile(input_forward_kernel);

//         init_inference_input_shader = lc::device.compile(init_inference_input_kernel);
//         fetch_inference_output_shader = lc::device.compile(fetch_inference_output_kernel);
//     }

//     void init() {
//         init_shader();
//         init_buffer();
//         refresh_weights();

//         lc::stream << init_inference_input_shader(inference_input).dispatch(inference_input.size());
//     }

//     vector<half> tmp(network::weight_size);
//     void refresh_weights() {
//         int i = 0;
//         for (half &x: tmp) {
//             x = network::weight_buffer[i];
//             i++;
//         }

//         lc::stream << weight_buffer.copy_from(tmp.data());
//     }

//     void inference() {
//         refresh_weights();
//         lc::stream << input_forward_shader(inference_image_size, input_weight, inference_input, inference_tmp).dispatch(inference_image_size / 2);
//         for (int i = 0; i < network::hidden_layers - 1; i++) {
//             lc::stream << layer_forward_shader(inference_image_size, layer_weight[i], inference_tmp, inference_tmp).dispatch(inference_image_size / 2);
//         }
//         lc::stream << output_forward_shader(inference_image_size, output_weight, inference_tmp, inference_output).dispatch(inference_image_size / 4);
//         lc::stream << fetch_inference_output_shader(inference_output, inference_image).dispatch(inference_size);
//     }
// }