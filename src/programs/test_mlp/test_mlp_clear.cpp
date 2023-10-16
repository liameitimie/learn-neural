#include <luisa/luisa-compute.h>
#include <linear_layer.h>
#include <frequency_encode_layer.h>
#include <ngp_encode_layer.h>
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

const int input_dim = 2;
const int output_dim = 3;
const int hidden_layers = 3;
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

void init_layers() {
    // layers.push_back(new FrequencyEncodeLayer(input_dim, layer_width));
    layers.push_back(new NGPEncodeLayer(input_dim, layer_width, 1<<19));
    for (int i = 0; i < hidden_layers - 1; i++) {
        layers.push_back(new LinearLayer(layer_width, layer_width, true, activation::Sine, weight_scale_siren(layer_width)));
    }
    layers.push_back(new LinearLayer(layer_width, output_dim, true, activation::None, weight_scale_siren(layer_width)));
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
}

void train(BufferView<half4> input, BufferView<half4> target, Buffer<half4> *train_loss) {
    int n = layers.size();

    layers[0]->forward(input, fwd_tmp[0]);
    for (int i = 1; i < n; i++) {
        layers[i]->forward(fwd_tmp[i-1], fwd_tmp[i]);
    }

    loss.evaluate(train_batch_size, fwd_tmp[n-1], target, train_loss?*train_loss:BufferView<half4>(), bwd_tmp[n-1]);
    for (int i = n-1; i > 0; i--) {
        layers[i]->backward(fwd_tmp[i-1], fwd_tmp[i], bwd_tmp[i], bwd_tmp[i-1], arenas[i]?arenas[i]:BufferView<half4>());
    }
    layers[0]->backward(input, fwd_tmp[0], bwd_tmp[0], BufferView<half4>(), arenas[0]?arenas[0]:BufferView<half4>());

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

    // init_inference_input(input);
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

    Window window{"", 1920, 1080};
    Swapchain swapchain = global::device().create_swapchain(
        window.native_handle(),
        global::stream(),
        make_uint2(1920, 1080),
        false, false,
        3
    );

    vector<half> tmp(output_dim*train_batch_size);
    vector<half> tmp1(32);
    vector<half> tmp2(32);
    vector<half> tmp3(32);

    while(!window.should_close()) {
        prepare_train_data(train_input, train_target);
        train(train_input, train_target, &train_loss);
        inference();
        // global::stream()
        //     << global::cmd_list().commit()
        //     << train_loss.copy_to(tmp.data())
        //     << train_target.view(0, 8).copy_to(tmp1.data())
        //     << fwd_tmp.back().view(0, 8).copy_to(tmp2.data())
        //     << output.view(0, 8).copy_to(tmp3.data())
        //     << synchronize();
        
        // static uint step = 0;
        // float x = 0;
        // for (float y: tmp) x += y;
        // print("step: {}, loss: {}\n", step, x);
        // step++;

        // for (float x: tmp1) print("{}, ", x);
        // print("\n");
        // for (float x: tmp2) print("{}, ", x);
        // print("\n");
        // for (float x: tmp3) print("{}, ", x);
        // print("\n\n");
        global::stream() << global::cmd_list().commit();
        // break;
        global::stream() << swapchain.present(inference_image);
        window.poll_events();
    }
    global::stream().synchronize();

    // vector<half> tmp(64);
    // global::stream() << ((NGPEncodeLayer*)layers[0])->feature_gradient().view(0, 32).copy_to(tmp.data()) << synchronize();
    // for (float x: tmp) print("{}, ", x);
    return 0;
}