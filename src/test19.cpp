#include <luisa/luisa-compute.h>
#include <luisa/gui/window.h>
#include <stb/stb_image.h>
#include <iostream>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

#define use_output_bias 1
#define use_layer_bias 1
#define use_input_encode 1
#define pi 3.1415926535897

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
const uint output_dim = 3;
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
    float learning_rate = 0.01;
    float beta1 = 0.9;
    float beta2 = 0.99;
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

namespace encoder {
    const uint encode_width = 32;
    const uint L = encode_width / input_dim;

    void encode(mat &input, mat &fwd_tmp) {
        const uint batch_size = input.cols;
        for (int t = 0; t < batch_size; t++) {
            for (int i = 0; i < input_dim; i++) {
                float x = input(i, t);
                // for (int l = 0; l < L; l++) {
                //     fwd_tmp(l + i*L, t) = sin(x * (1 << l) * pi);
                // }
                for (int l = 0; l < L / 2; l++) {
                    fwd_tmp(l*2 + i*L, t) = sin(x * (1 << l) * pi);
                    fwd_tmp(l*2 + 1 + i*L, t) = cos(x * (1 << l) * pi);
                }
            }
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
    
    float* gradient_buffer;
    mat layer_weight_gradient[hidden_layers - 1];
    mat input_weight_gradient;
    mat output_weight_gradient;

#if use_layer_bias
    mat layer_bias[hidden_layers];
    mat layer_bias_gradient[hidden_layers];
#endif

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
        for (int i = 0; i < v.rows*v.cols; i++) {
            activation_forward_host(v.data[i]);
        }
    }
    void apply_backward_activation(mat &g, mat &act) {
        assert(g.rows == act.rows && g.cols == act.cols && g.layout == act.layout);
        if (g.rows != act.rows || g.cols != act.cols || g.layout != act.layout) {
            fmt::print("error apply_backward_activation\n");
            exit(0);
        }
        for (int i = 0; i < g.rows*g.cols; i++) {
            activation_backward_host(g.data[i], act.data[i]);
        }
    }

    void calc_output_gradiant(uint batch_size, mat &output, mat &target, mat &output_grad, mat &output_loss) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < output_dim; j++) {
                output_loss(j, i) = prediction_loss_host(output(j, i), target(j, i), batch_size);
                output_grad(j, i) = loss_gradient_host(output(j, i), target(j, i), batch_size);
            }
        }
    }

    void apply_layer_bias(uint batch_size, mat &fwd_tmp, int layer_id) {
#if use_layer_bias
        for (int i = 0; i < layer_width; i++) {
            for (int j = 0; j < batch_size; j++) {
                fwd_tmp(i, j) += layer_bias[layer_id](i, 0);
            }
        }
#endif
    }
    void calc_layer_bias_gradient(uint batch_size, mat &bwd_tmp, int layer_id) {
#if use_layer_bias
        for (int i = 0; i < layer_bias_gradient[layer_id].size; i++) {
            layer_bias_gradient[layer_id].data[i] = 0;
        }
        // print("bwd_tmp", bwd_tmp);
        for (int i = 0; i < layer_width; i++) {
            for (int j = 0; j < batch_size; j++) {
                layer_bias_gradient[layer_id](i, 0) += bwd_tmp(i, j);
            }
        }
#endif
    }
    void apply_output_bias(uint batch_size, mat& output) {
#if use_output_bias
        for (int j = 0; j < batch_size; j++) {
            for (int i = 0; i < output_dim; i++) {
                output(i, j) += output_bias(i, 0);
            }
        }
#endif
    }
    void calc_output_bias_gradient(uint batch_size, mat &output_grad) {
#if use_output_bias
        for (int i = 0; i < output_bias_gradient.size; i++) {
            output_bias_gradient.data[i] = 0;
        }
        for (int i = 0; i < output_dim; i++) {
            for (int j = 0; j < batch_size; j++) {
                output_bias_gradient(i, 0) += output_grad(i, j);
            }
        }
#endif
    }

    void train_step(uint batch_size, mat &input, mat &target, mat &output, mat &output_grad, mat &output_loss, mat *fwd_tmp, mat *bwd_tmp) {
        const uint n = hidden_layers;

        // forward
#if use_input_encode
        encoder::encode(input, fwd_tmp[n]);
        matmul(input_weight, fwd_tmp[n], fwd_tmp[0]);
#else
        matmul(input_weight, input, fwd_tmp[0]);
#endif
        apply_layer_bias(batch_size, fwd_tmp[0], 0);
        apply_forward_activation(fwd_tmp[0]);

        for (int i = 0; i < n - 1; i++) {
            matmul(layer_weight[i], fwd_tmp[i], fwd_tmp[i + 1]);
            apply_layer_bias(batch_size, fwd_tmp[i + 1], i + 1);
            apply_forward_activation(fwd_tmp[i + 1]);
        }
        matmul(output_weight, fwd_tmp[n - 1], output);
        apply_output_bias(batch_size, output);

        calc_output_gradiant(batch_size, output, target, output_grad, output_loss);

        // backward
        for (int i = 0; i < n - 1; i++) layer_weight[i].transpose();
        input_weight.transpose();
        output_weight.transpose();

        matmul(output_weight, output_grad, bwd_tmp[n - 1]);
        apply_backward_activation(bwd_tmp[n - 1], fwd_tmp[n - 1]);
        for (int i = n - 1; i > 0; i--) {
            matmul(layer_weight[i - 1], bwd_tmp[i], bwd_tmp[i - 1]);
            apply_backward_activation(bwd_tmp[i - 1], fwd_tmp[i - 1]);
        }

        for (int i = 0; i < n - 1; i++) layer_weight[i].transpose();
        input_weight.transpose();
        output_weight.transpose();

        // calc weight gradient
        input.transpose();
        for (int i = 0; i < n + 1; i++) fwd_tmp[i].transpose();

#if use_input_encode
        matmul(bwd_tmp[0], fwd_tmp[n], input_weight_gradient);
#else
        matmul(bwd_tmp[0], input, input_weight_gradient);
#endif
        for (int i = 0; i < n - 1; i++) {
            matmul(bwd_tmp[i + 1], fwd_tmp[i], layer_weight_gradient[i]);
        }
        matmul(output_grad, fwd_tmp[n - 1], output_weight_gradient);

        input.transpose();
        for (int i = 0; i < n + 1; i++) fwd_tmp[i].transpose();

        calc_output_bias_gradient(batch_size, output_grad);
        for (int i = 0; i < hidden_layers; i++) {
            calc_layer_bias_gradient(batch_size, bwd_tmp[i], i);
        }

        optimizer::optimize(weight_buffer, gradient_buffer);
    }

    void inference(uint batch_size, mat &input, mat &output, mat *fwd_tmp) {
        const uint n = hidden_layers;
#if use_input_encode
        encoder::encode(input, fwd_tmp[1]);
        matmul(input_weight, fwd_tmp[1], fwd_tmp[0]);
#else
        matmul(input_weight, input, fwd_tmp[0]);
#endif
        apply_layer_bias(batch_size, fwd_tmp[0], 0);
        apply_forward_activation(fwd_tmp[0]);
        for (int i = 0; i < n - 1; i++) {
            matmul(layer_weight[i], fwd_tmp[i & 1], fwd_tmp[~i & 1]);
            apply_layer_bias(batch_size, fwd_tmp[~i & 1], i + 1);
            apply_forward_activation(fwd_tmp[~i & 1]);
        }
        matmul(output_weight, fwd_tmp[(n - 1)&1], output);
        apply_output_bias(batch_size, output);
    }

    
}

namespace trainer {
    const uint train_batch_size = 16384;
#if use_input_encode
    mat fwd_tmp[network::hidden_layers + 1];
#else
    mat fwd_tmp[network::hidden_layers];
#endif
    mat bwd_tmp[network::hidden_layers];
    mat output;
    mat output_grad;
    mat output_loss;
    mat input;
    mat target;

    img train_image;

    const uint2 inference_res = uint2{192, 108};
    const uint inference_size = inference_res.x * inference_res.y;

    Image<float> gpu_inference_image;
    stbi_uc* inference_image;
    mat inference_input;
    mat inference_output;
    mat inference_tmp[2];

    void init_buffer();
    void init(const char* file) {
        train_image.load(file);
        init_buffer();
        network::init();
    }

    uint t = 0;

    void prepare_train_data() {
        for (int i = 0; i < train_batch_size; i++) {
            float2 s = sobol_2d(233 + i + t*train_batch_size);
            float4 c = train_image.read(s);
            for (int j = 0; j < 2; j++) input(j, i) = s[j];
            for (int j = 0; j < 3; j++) target(j, i) = c[j];
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
        print("step:{}, loss: {}\n", t, loss);
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
                    inference_image[j + i*4] = clamp((int)round(255 * inference_output(j, i)), 0, 255);
                }
            }
        }
        lc::stream << gpu_inference_image.copy_from(inference_image);
    }
}

int main(int argc, char *argv[]) {
    lc::init(argv[0]);

    trainer::init("assets/nahida.jpeg");

    Window window{"", trainer::inference_res};
    Swapchain swapchain = lc::device.create_swapchain(
        window.native_handle(),
        lc::stream,
        trainer::inference_res,
        false, false,
        3
    );

    while(!window.should_close()){
        trainer::train_step();
        trainer::inference();
        lc::stream << swapchain.present(trainer::gpu_inference_image);
        window.poll_events();
    }
    lc::stream.synchronize();

    return 0;
}

namespace network {
    
    const uint layer_weight_size = layer_width * layer_width;
#if use_input_encode
    const uint input_weight_size = encoder::encode_width * layer_width;
#else
    const uint input_weight_size = input_dim * layer_width;
#endif
    const uint output_weight_size = output_dim * layer_width;

    const uint layer_bias_size = layer_width;
    const uint output_bias_size = output_dim;

    const uint weight_size = 
        layer_weight_size * (hidden_layers - 1)
        + input_weight_size
        + output_weight_size
#if use_layer_bias
        + layer_bias_size * hidden_layers
#endif
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
#if use_input_encode
        input_weight = mat(layer_width, encoder::encode_width, ColMajor, weight_buffer + offset);
        input_weight_gradient = mat(layer_width, encoder::encode_width, ColMajor, gradient_buffer + offset);
#else
        input_weight = mat(layer_width, input_dim, ColMajor, weight_buffer + offset);
        input_weight_gradient = mat(layer_width, input_dim, ColMajor, gradient_buffer + offset);
#endif
        offset += input_weight_size;

        output_weight = mat(output_dim, layer_width, RowMajor, weight_buffer + offset);
        output_weight_gradient = mat(output_dim, layer_width, RowMajor, gradient_buffer + offset);
        offset += output_weight_size;
#if use_layer_bias
        for (int i = 0; i < hidden_layers; i++) {
            layer_bias[i] = mat(layer_width, 1, ColMajor, weight_buffer + offset);
            layer_bias_gradient[i] = mat(layer_width, 1, ColMajor, gradient_buffer + offset);
            offset += layer_bias_size;
        }
#endif
#if use_output_bias
        output_bias = mat(output_dim, 1, ColMajor, weight_buffer + offset);
        output_bias_gradient = mat(output_dim, 1, ColMajor, gradient_buffer + offset);
#endif
    }

    void init_weight() {
        float weight_scale[3];
        if (activation == Activation::Sine) {
            weight_scale[0] = sqrt(6.0 / layer_width);
#if use_input_encode
            weight_scale[1] = sqrt(6.0 / (encoder::encode_width + layer_width));
            // weight_scale[1] = 30.0 / encoder::encode_width;
#else
            weight_scale[1] = 30.0 / input_dim;
#endif
            weight_scale[2] = sqrt(6.0 / layer_width);
        }
        else {
            weight_scale[0] = sqrt(6.0 / (layer_width + layer_width));
#if use_input_encode
            weight_scale[1] = sqrt(6.0 / (encoder::encode_width + layer_width));
#else
            weight_scale[1] = sqrt(6.0 / (input_dim + layer_width));
#endif
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
#if use_layer_bias
        for (int t = 0; t < hidden_layers; t++) {
            for (int i = 0; i < layer_bias[t].size; i++) {
                layer_bias[t].data[i] = 0.1;
            }
        }
#endif
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
#if use_layer_bias
        fmt::print("layer bias {}: ", 0);
        layer_bias[0].print();
        fmt::print("\n");
#endif
        for (int i = 0; i < hidden_layers - 1; i++) {
            fmt::print("layer weight {}: ", i);
            layer_weight[i].print();
            fmt::print("\n");
#if use_layer_bias
            fmt::print("layer bias {}: ", i + 1);
            layer_bias[i + 1].print();
            fmt::print("\n");
#endif
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
#if use_layer_bias
        fmt::print("layer bias gradient {}: ", 0);
        layer_bias_gradient[0].print();
        fmt::print("\n");
#endif
        for (int i = 0; i < hidden_layers - 1; i++) {
            fmt::print("layer weight gradient {}: ", i);
            layer_weight_gradient[i].print();
            fmt::print("\n");
#if use_layer_bias
            fmt::print("layer bias gradient {}: ", i + 1);
            layer_bias_gradient[i + 1].print();
            fmt::print("\n");
#endif
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
        for (int i = 0; i < network::hidden_layers; i++) {
            fwd_tmp[i] = mat(network::layer_width, train_batch_size, RowMajor);
        }
#if use_input_encode
        fwd_tmp[network::hidden_layers] = mat(encoder::encode_width, train_batch_size, RowMajor);
#endif
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
            inference_input(0, i) = (x + 0.5) / inference_res.x;
            inference_input(1, i) = (y + 0.5) / inference_res.y;
        }

        gpu_inference_image = lc::device.create_image<float>(PixelStorage::BYTE4, trainer::inference_res);
    }

}
