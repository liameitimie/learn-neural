#include <luisa/luisa-compute.h>
#include <linear_layer.h>
#include <frequency_encode_layer.h>
#include <l2.h>
#include <global.h>
#include <luisa/gui/window.h>
#include <stb/stb_image.h>
#include <gpu_rands.h>
#include <activation_func.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

#define PI 3.14159265358979323846f
#define test_with_cpu_layer 0

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

class CPULayer {
public:
    virtual void forward(mat &input, mat &output) {}
    virtual void backward(mat &fwd_input, mat &fwd_output, mat &output_grad, mat &input_grad) {}
};

class CPUFrequencyEncodeLayer: public CPULayer {
public:
    FrequencyEncodeLayer* layer;

    CPUFrequencyEncodeLayer(DiffLayer* _layer): layer((FrequencyEncodeLayer*)_layer) {

    }

    virtual void forward(mat &input, mat &output) override {
        int l = layer->output_dim() / layer->input_dim();
        for (int i = 0; i < layer->input_dim(); i++) {
            for (int p = 0; p < l; p++) {
                for (int j = 0; j < input.cols; j++) {
                    output(i*l+p, j) = sin(input(i, j) * ((1 << (p/2)) * PI) + (p%2)*(PI/2));
                }
            }
        }
    }
};

class CPULinearLayer: public CPULayer {
public:
    LinearLayer *layer;
    mat weight;
    mat bias;
    mat weight_grad;
    mat bias_grad;

    CPULinearLayer(DiffLayer *_layer): layer((LinearLayer*)_layer) {
        weight = mat(layer->output_dim(), layer->input_dim(), ColMajor);
        weight_grad = mat(layer->output_dim(), layer->input_dim(), ColMajor);
        if (layer->bias()) {
            bias = mat(layer->output_dim(), 1 ,ColMajor);
            bias_grad = mat(layer->output_dim(), 1 ,ColMajor);
        }

        int pad_out_dim = (layer->output_dim()+3)/4*4;
        vector<half> tmp1(pad_out_dim*layer->input_dim());
        vector<half> tmp2(pad_out_dim);

        global::stream() << global::cmd_list().commit() << synchronize();
        global::stream() << layer->weight().copy_to(tmp1.data());
        if (layer->bias()) {
            global::stream() << layer->bias().copy_to(tmp2.data());
        }
        global::stream().synchronize();

        if (layer->bias()) {
            for (int i = 0; i < layer->output_dim(); i++) {
                bias(i, 0) = tmp2[i];
            }
        }
        for (int i = 0; i < layer->input_dim(); i++) {
            for (int j = 0; j < layer->output_dim(); j++) {
                weight(j, i) = tmp1[j + i*pad_out_dim];
            }
        }
    }
    virtual void forward(mat &input, mat &output) override {
        matmul(weight, input, output);
        if (layer->bias()) {
            for (int i = 0; i < layer->input_dim(); i++) {
                for (int j = 0; j < output.cols; j++) {
                    output(j, i) += bias(j, 0);
                }
            }
        }
        if (layer->activation() != activation::None) {
            for (int i = 0; i < layer->input_dim(); i++) {
                for (int j = 0; j < output.cols; j++) {
                    output(i, j) = activation::forward_host(layer->activation(), output(i, j));
                }
            }
        }
    }
    virtual void backward(mat &fwd_input, mat &fwd_output, mat &output_grad, mat &input_grad) override {
        if (layer->activation() != activation::None) {
            for (int i = 0; i < output_grad.size; i++) {
                output_grad.data[i] = activation::backward_host(layer->activation(), output_grad.data[i], fwd_output.data[i]);
            }
        }
        weight.transpose();
        matmul(weight, output_grad, input_grad);
        weight.transpose();

        if (layer->bias()) {
            for (int i = 0; i < bias_grad.size; i++) bias_grad.data[i] = 0;
            for (int i = 0; i < layer->output_dim(); i++) {
                for (int j = 0; j < output_grad.cols; j++) {
                    bias_grad(i, 0) += output_grad(i, j);
                }
            }
        }
        fwd_input.transpose();
        matmul(output_grad, fwd_input, weight_grad);
        fwd_input.transpose();
    }
};

const int input_dim = 2;
const int output_dim = 3;
const int hidden_layers = 2;
const int layer_width = 32;
const int train_batch_size = 16384;
const int pixel_size = 1920*1080;

Image<float> image;
BindlessArray heap;

void load_image(const char* file) {
    int width = 0;
    int height = 0;
    int channel = 0;
    auto pixel = stbi_load(file, &width, &height, &channel, 4);

    image = global::device().create_image<float>(PixelStorage::BYTE4, width, height);
    heap = global::device().create_bindless_array(1);

    global::stream() << image.copy_from(pixel) << synchronize();

    heap.emplace_on_update(0, image, Sampler::linear_linear_mirror());
    global::stream() << heap.update() << synchronize();
}

Shader1D<BindlessArray, Buffer<half4>, Buffer<half4>, uint> prepare_train_data_shader;

void prepare_train_data(BufferView<half4> train_input, BufferView<half4> train_target) {
    if (!prepare_train_data_shader) {
        Kernel1D prepare_train_data_kernel = []($bindless heap, $buffer<half4> input, $buffer<half4> target, $uint ofs) {
            set_block_size(256);
            $uint tid = $dispatch_x;
            $float2 s[4];
            $float4 c[4];
            for (int i = 0; i < 4; i++) {
                s[i] = sobol_2d(i + tid*4 + ofs);
                c[i] = heap.tex2d(0).sample(s[i]);
            }
            $half4 in[4];
            $half4 out[4];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < input_dim; j++) in[j][i] = s[i][j];
                for (int j = 0; j < output_dim; j++) out[j][i] = c[i][j];
            }
            for (int i = 0; i < input_dim; i++) input.write(tid + i*train_batch_size/4, in[i]);
            for (int i = 0; i < output_dim; i++) target.write(tid + i*train_batch_size/4, out[i]);
        };
        prepare_train_data_shader = global::device().compile(prepare_train_data_kernel);
    }
    static uint t = 0;
    global::cmd_list() << prepare_train_data_shader(heap, train_input, train_target, 233 + t*train_batch_size).dispatch(train_batch_size/4);
    t++;
}

Shader1D<Buffer<half4>, uint> init_inference_input_shader;
Shader1D<Buffer<half4>, Image<float>> fetch_inference_output_shader;

void init_inference_input(BufferView<half4> input) {
    if (!init_inference_input_shader) {
        Kernel1D init_inference_input_kernel = []($buffer<half4> inference_input, $uint t) {
            $uint tid = $dispatch_x;
            $half2 uv[4];
            for (int i = 0; i < 4; i++) {
                $uint idx = tid*4 + i;
                $float2 ofs = sobol_2d(t);
                $uint x = idx % 1920;
                $uint y = idx / 1920;
                uv[i][0] = (x + ofs.x) / 1920;
                uv[i][1] = (y + ofs.y) / 1080;
            }
            $half4 tmp[4];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < input_dim; j++) tmp[j][i] = uv[i][j];
            }
            for (int i = 0; i < input_dim; i++) {
                inference_input.write(tid + i*pixel_size/4, tmp[i]);
            }
        };
        init_inference_input_shader = global::device().compile(init_inference_input_kernel);
    }
    static uint t = 0;
    t++;
    global::cmd_list() << init_inference_input_shader(input, t).dispatch(pixel_size/4);
}
void fetch_inference_output(Buffer<half4> &output, Image<float> &image) {
    if (!fetch_inference_output_shader) {
        Kernel1D fetch_inference_output_kernel = []($buffer<half4> inference_output, $image<float> inference_image) {
            $uint tid = $dispatch_x;
            $half4 out[4];
            for (int i = 0; i < output_dim; i++) {
                out[i] = inference_output.read(tid + i*pixel_size/4);
            }
            $float4 c[4];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < output_dim; j++) c[i][j] = out[j][i];
            }
            for (int i = 0; i < 4; i++) {
                $uint idx = tid*4 + i;
                $uint x = idx % 1920;
                $uint y = idx / 1920;
                inference_image.write($uint2{x, y}, c[i]);
            }
        };
        fetch_inference_output_shader = global::device().compile(fetch_inference_output_kernel);
    }
    global::cmd_list() << fetch_inference_output_shader(output, image).dispatch(pixel_size/4);
}

L2Loss loss;
vector<DiffLayer*> layers;
vector<Buffer<half4>> fwd_tmp;
vector<Buffer<half4>> bwd_tmp;
vector<Buffer<half4>> arenas;

vector<CPULayer*> cpu_layers;
vector<mat> cpu_fwd_tmp;
vector<mat> cpu_bwd_tmp;
mat cpu_input(input_dim, train_batch_size, RowMajor);
mat cpu_target(output_dim, train_batch_size, RowMajor);
mat cpu_loss(output_dim, train_batch_size, RowMajor);

void calc_cpu_loss() {
    for (int i = 0; i < output_dim; i++) {
        for (int j = 0; j < train_batch_size; j++) {
            float s = cpu_fwd_tmp.back()(i, j) - cpu_target(i, j);
            cpu_loss(i, j) = s*s/train_batch_size;
            cpu_bwd_tmp.back()(i, j) = 2*s/train_batch_size;
        }
    }
}

void init_layers() {
    // layers.push_back(new LinearLayer(input_dim, layer_width, true, activation::Sine, 100));
    layers.push_back(new FrequencyEncodeLayer(input_dim, layer_width));
    for (int i = 0; i < hidden_layers - 1; i++) {
        layers.push_back(new LinearLayer(layer_width, layer_width, true, activation::Sine, weight_scale_xavier(layer_width, layer_width)));
    }
    layers.push_back(new LinearLayer(layer_width, output_dim, true, activation::None, weight_scale_xavier(layer_width, output_dim)));

#if test_with_cpu_layer
    cpu_layers.push_back(new CPUFrequencyEncodeLayer(layers[0]));
    for (int i = 0; i < hidden_layers - 1; i++) {
        cpu_layers.push_back(new CPULinearLayer(layers[i+1]));
    }
    cpu_layers.push_back(new CPULinearLayer(layers[hidden_layers]));
#endif
}

void prepare_network() {
    int n = layers.size();

    fwd_tmp.resize(n);
    for (int i = 0; i < n; i++) {
        if (i < n-1 && layers[i]->output_dim() != layers[i+1]->input_dim()) {
            print("layer{} output_dim {} not match layer{} input_dim {}\n", i, i+1, layers[i]->output_dim(), layers[i+1]->input_dim());
            exit(0);
        }
        fwd_tmp[i] = global::device().create_buffer<half4>(layers[i]->output_dim() * train_batch_size/4);
    }
    bwd_tmp.resize(n);
    for (int i = 0; i < n; i++) {
        bwd_tmp[i] = global::device().create_buffer<half4>(layers[i]->output_dim() * train_batch_size/4);
    }
    arenas.resize(n);
    for (int i = 0; i < n; i++) {
        int size = layers[i]->arena_size(train_batch_size) / 4;
        if (size) {
            arenas[i] = global::device().create_buffer<half4>(size);
        }
    }
#if test_with_cpu_layer
    cpu_fwd_tmp.resize(n);
    for (int i = 0; i < n; i++) {
        cpu_fwd_tmp[i] = mat(layers[i]->output_dim(), train_batch_size, RowMajor);
    }
    cpu_bwd_tmp.resize(n);
    for (int i = 0; i < n; i++) {
        cpu_bwd_tmp[i] = mat(layers[i]->output_dim(), train_batch_size, RowMajor);
    }
#endif
}

mat ttttmp(input_dim, train_batch_size, RowMajor);

vector<half> ttmp(layer_width*train_batch_size);

void init_cpu_train(BufferView<half4> input, BufferView<half4> target) {
    global::stream() << global::cmd_list().commit() << synchronize();
    global::stream() << input.copy_to(ttmp.data()) << synchronize();
    for (int i = 0; i < input_dim; i++) {
        for (int j = 0; j < train_batch_size; j++) {
            cpu_input(i, j) = ttmp[j + i*train_batch_size];
        }
    }
    global::stream() << target.copy_to(ttmp.data()) << synchronize();
    for (int i = 0; i < output_dim; i++) {
        for (int j = 0; j < train_batch_size; j++) {
            cpu_target(i, j) = ttmp[j + i*train_batch_size];
        }
    }
}

void print_train_data() {
    print("train_data:\n");
    for (int i = 0; i < 32; i++) {
        print("(");
        for (int j = 0; j < input_dim; j++) {
            print("{}", cpu_input(j, i));
            if (j < input_dim-1) print(", ");
        }
        print(") -> (");
        for (int j = 0; j < output_dim; j++) {
            print("{}", cpu_target(j, i));
            if (j < output_dim-1) print(", ");
        }
        print(")\n");
    }
}

void compare(BufferView<half4> d, mat h, string name) {
    global::stream() << global::cmd_list().commit() << synchronize();
    global::stream() << d.copy_to(ttmp.data()) << synchronize();

    auto trans_idx = [&](int idx) {
        if (h.layout == ColMajor) {
            if (h.rows < 4) {
                return idx/h.rows*4 + idx%h.rows;
            }
        }
        return idx;
    };

    print("\n{}:\n", name);
    print("h: [");
    for (int i = 0; i < min(32, h.size); i++) {
        print("{}, ", h.data[i]);
    }
    print("]\n");

    print("d: [");
    for (int i = 0; i < min(32, h.size); i++) {
        print("{}, ", ttmp[trans_idx(i)]);
    }
    print("]\n");

    float f_err = 0;
    int err_c = 0;
    for (int i = 0; i < h.size; i++) {
        float t1 = h.data[i];
        float t2 = ttmp[trans_idx(i)];
        float err = abs(t1 - t2);
        f_err = max(f_err, err);
        if (err > 0.01) {
            if (err_c < 32) {
                print("error {}: {}, {}\n", i, t1, t2);
            }
            err_c++;
        }
    }
    print("f_err: {}\n", f_err);
    print("err_c: {}\n", err_c);
    print("ok\n\n");
}

void train(BufferView<half4> input, BufferView<half4> target, Buffer<half4> *train_loss) {
    int n = layers.size();

    layers[0]->forward(input, fwd_tmp[0]);
#if test_with_cpu_layer
    cpu_layers[0]->forward(cpu_input, cpu_fwd_tmp[0]);
    compare(fwd_tmp[0], cpu_fwd_tmp[0], "fwd_tmp[0]");
#endif
    for (int i = 1; i < n; i++) {
        layers[i]->forward(fwd_tmp[i-1], fwd_tmp[i]);
#if test_with_cpu_layer
        cpu_layers[i]->forward(cpu_fwd_tmp[i-1], cpu_fwd_tmp[i]);
        compare(fwd_tmp[i], cpu_fwd_tmp[i], format("fwd_tmp[{}]", i));
#endif
    }

    loss.evaluate(train_batch_size, fwd_tmp[n-1], target, train_loss?*train_loss:BufferView<half4>(), bwd_tmp[n-1]);
#if test_with_cpu_layer
    calc_cpu_loss();
    compare(*train_loss, cpu_loss, "train_loss");
    compare(bwd_tmp[n-1], cpu_bwd_tmp[n-1], "output_grad");
#endif

    for (int i = n-1; i > 0; i--) {
        layers[i]->backward(fwd_tmp[i-1], fwd_tmp[i], bwd_tmp[i], bwd_tmp[i-1], arenas[i]?arenas[i]:BufferView<half4>());
#if test_with_cpu_layer
        cpu_layers[i]->backward(cpu_fwd_tmp[i-1], cpu_fwd_tmp[i], cpu_bwd_tmp[i], cpu_bwd_tmp[i-1]);
        compare(bwd_tmp[i-1], cpu_bwd_tmp[i-1], format("bwd_tmp[{}]", i-1));
        compare(((LinearLayer*)layers[i])->weight_grad(), ((CPULinearLayer*)cpu_layers[i])->weight_grad, format("layer[{}].weight_grad", i));
        if (((LinearLayer*)layers[i])->bias()) {
            compare(((LinearLayer*)layers[i])->bias_grad(), ((CPULinearLayer*)cpu_layers[i])->bias_grad, format("layer[{}].bias_grad", i));
        }
#endif
    }

    // layers[0]->backward(input, fwd_tmp[0], bwd_tmp[0], BufferView<half4>(), arenas[0]?arenas[0]:BufferView<half4>());
    // cpu_layers[0]->backward(cpu_input, cpu_fwd_tmp[0], cpu_bwd_tmp[0], ttttmp);

    for (int i = 0; i < n; i++) {
        layers[i]->optimize();
    }
}

int main(int argc, char** argv) {
    global::init(argv[0]);
    load_image("assets/nahida.jpeg");

    init_layers();
    prepare_network();

    auto train_input = global::device().create_buffer<half4>(input_dim * train_batch_size / 4);
    auto train_target = global::device().create_buffer<half4>(output_dim * train_batch_size / 4);
    auto train_loss = global::device().create_buffer<half4>(output_dim * train_batch_size / 4);

    auto input = global::device().create_buffer<half4>(input_dim * pixel_size / 4);
    auto output = global::device().create_buffer<half4>(output_dim * pixel_size / 4);
    auto inference_tmp = global::device().create_buffer<half4>(layer_width * pixel_size / 4);
    auto inference_image = global::device().create_image<float>(PixelStorage::BYTE4, 1920, 1080);

    auto inference = [&]() {
        init_inference_input(input);
        int n = layers.size();
        layers[0]->forward(input, inference_tmp);
        for (int i = 1; i < n-1; i++) {
            layers[i]->forward(inference_tmp, inference_tmp);
        }
        layers[n-1]->forward(inference_tmp, output);
        fetch_inference_output(output, inference_image);
    };

    // for (int i = 0; i < 2; i++) {
    //     prepare_train_data(train_input, train_target);
    //     init_cpu_train(train_input, train_target);
    //     print_train_data();
    //     // train(train_input, train_target, &train_loss);
    // }
    // global::stream() << global::cmd_list().commit() << synchronize();

    Window window{"", 1920, 1080};
    Swapchain swapchain = global::device().create_swapchain(
        window.native_handle(),
        global::stream(),
        make_uint2(1920, 1080),
        false, false,
        3
    );

    // vector<half> tmp(output_dim*train_batch_size);

    while(!window.should_close()) {
        prepare_train_data(train_input, train_target);
        train(train_input, train_target, &train_loss);
        inference();
        // global::stream()
        //     << global::cmd_list().commit()
        //     << train_loss.copy_to(tmp.data())
        //     << synchronize();
        
        // static uint step = 0;
        // float x = 0;
        // for (float y: tmp) x += y;
        // print("step: {}, loss: {}\n", step, x);
        // step++;
        global::stream() << global::cmd_list().commit();
        global::stream() << swapchain.present(inference_image);
        window.poll_events();
    }
    global::stream().synchronize();

    return 0;
}