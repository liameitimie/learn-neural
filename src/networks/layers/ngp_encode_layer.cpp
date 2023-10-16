#include <luisa/dsl/sugar.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>
#include <ngp_encode_layer.h>
#include <global.h>
#include <gpu_rands.h>

using namespace luisa;
using namespace luisa::compute;

const int N_min = 16;

int powi(int x, int p) {
    int res = 1;
    for (int i = 0; i < p; i++) {
        res *= x;
    }
    return res;
}
$uint powi($uint x, int p) {
    $uint res = 1;
    for (int i = 0; i < p; i++) {
        res *= x;
    }
    return res;
}

NGPEncodeLayer::NGPEncodeLayer(int input_dim, int output_dim, int max_level_table_size, int levels, AdamConfig optim_cfg):
    DiffLayer(input_dim, output_dim),
    optim(optim_cfg),
    max_level_table_size(max_level_table_size),
    levels(levels)
    // feature_per_level(feature_per_level)
{
    if (input_dim > 4) {
        fmt::print("error: ngp_encoder not impl for input_dim > 4\n");
        exit(1);
    }
    if (levels * feature_per_level > output_dim) {
        fmt::print("error: feature num > output_dim\n");
        exit(1);
    }

    init_level_offset();
    _feature_table = global::device().create_buffer<half2>(table_size);
    _feature_gradient = global::device().create_buffer<half2>(table_size);

    reset_parameters();
    optim.init(_feature_table.view().as<half4>());
}

void NGPEncodeLayer::init_level_offset() {
    level_offset.resize(levels + 1);
    int offset = 0;
    for (int i = 0; i < levels; i++) {
        int res = (1 << i) * N_min;
        int level_size = 0;
        if (pow(res, input_dim()) > max_level_table_size + 50) {
            level_size = max_level_table_size;
        }
        else {
            level_size = min(powi(res, input_dim()), max_level_table_size);
        }
        level_offset[i] = offset;
        offset += level_size;
    }
    level_offset[levels] = offset;
    table_size = offset;

    level_offset_buffer = global::device().create_buffer<int>(levels + 1);
    global::stream() << level_offset_buffer.copy_from(level_offset.data());
}

void trans_input($float4 *in, int input_dim) {
    $float tmp;
    auto swap = [&]($float &a, $float &b) {
        tmp = a;
        a = b;
        b = tmp;
    };
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < min(input_dim, j); i++) {
            swap(in[i][j], in[j][i]);
        }
    }
}
void calc_pos($float4 &in, $float4 &pos, $uint4 &grid_idx, $uint &grid_res) {
    pos = in * (grid_res - 1).cast<float>();
    $float4 tmp = floor(pos);
    grid_idx = tmp;
    pos -= tmp;
}
$uint table_idx(int dim, $uint4 &grid_idx, $uint &grid_res, $uint &level_size, $bool &use_hash) {
    $uint idx = 0;
    $if (use_hash) {
        uint prime[4] = {1u, 2654435761u, 805459861u, 3674653429u};
        for (int i = 0; i < dim; i++) {
            idx ^= grid_idx[i] * prime[i];
        }
    }
    $else {
        $uint s = 1;
        for (int i = 0; i < dim; i++) {
            idx += grid_idx[i] * s;
            s *= grid_res;
        }
    };
    return idx % level_size;
}

void ngp_encode_kernel_impl(
    $uint &batch_size, 
    int input_dim, 
    $buffer<half4> &input, 
    $buffer<half4> &output, 
    $buffer<half2> &feature_table, 
    $buffer<int> &level_offsets
) {
    set_block_size(256, 1);
    $uint tid = $dispatch_x;
    $uint level = $dispatch_y;
    $uint level_offset = level_offsets.read(level);
    $uint level_size = level_offsets.read(level + 1) - level_offset;

    $uint grid_res = (1u << level) * N_min;
    $bool use_hash = (pow(grid_res.cast<float>(), (float)input_dim) > level_size + 50) | (powi(grid_res, input_dim) > level_size);

    $float4 in[4];
    $float4 pos[4];
    $uint4 grid_idx[4];

    for (int i = 0; i < input_dim; i++) {
        in[i] = input.read(tid + i*batch_size/4);
    }
    trans_input(in, input_dim);
    for (int i = 0; i < 4; i++) {
        calc_pos(in[i], pos[i], grid_idx[i], grid_res);
    }

    $float2 res_feature[4];
    $float w;
    $uint4 idx;
    $float2 feature;
    for (int i = 0; i < 4; i++) {
        for (int t = 0; t < powi(2, input_dim); t++) {
            w = 1;
            for (int d = 0; d < input_dim; d++) {
                if ((t & (1 << d)) == 0) {
                    w *= 1 - pos[i][d];
                    idx[d] = grid_idx[i][d];
                }
                else {
                    w *= pos[i][d];
                    idx[d] = grid_idx[i][d] + 1;
                }
            }
            feature = feature_table.read(level_offset + table_idx(input_dim, idx, grid_res, level_size, use_hash));
            // feature = feature_table.read(level_offset + tea(tid, t*4+i).x % level_size);
            res_feature[i] += w * feature;
        }
    }
    $half4 tmp[2];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            tmp[i][j] = res_feature[j][i];
        }
    }
    output.write(tid + level*2*batch_size/4, tmp[0]);
    output.write(tid + (level*2+1)*batch_size/4, tmp[1]);
}

Shader2D<uint, Buffer<half4>, Buffer<half4>, Buffer<half2>, Buffer<int>> ngp_encode_shader[5];

void NGPEncodeLayer::forward(const BufferView<half4> input, BufferView<half4> output) {
    if (!ngp_encode_shader[input_dim()]) {
        Kernel2D ngp_encode_kernel = [&](
            $uint batch_size,
            $buffer<half4> input, 
            $buffer<half4> output, 
            $buffer<half2> feature_table, 
            $buffer<int> level_offsets
        ) {
            ngp_encode_kernel_impl(batch_size, input_dim(), input, output, feature_table, level_offsets);
        };
        ngp_encode_shader[input_dim()] = global::device().compile(ngp_encode_kernel);
    }
    const uint batch_size = input.size()*4 / input_dim();
    global::cmd_list() << ngp_encode_shader[input_dim()](batch_size, input, output, _feature_table, level_offset_buffer).dispatch(batch_size/4, levels);
}

Shader1D<Buffer<half2>> init_feature_shader;

void NGPEncodeLayer::reset_parameters() {
    if (!init_feature_shader) {
        Kernel1D init_feature_kernel = []($buffer<half2> feature_table) {
            set_block_size(256);
            $uint tid = $dispatch_x;
            $half2 f;
            $uint2 s = tea(tid, 233);
            f.x = 1e-4f * as_uniform(s.x);
            f.y = 1e-4f * as_uniform(s.y);
            feature_table.write(tid, f);
        };
        init_feature_shader = global::device().compile(init_feature_kernel);
    }
    global::cmd_list() << init_feature_shader(_feature_table).dispatch(_feature_table.size());
}

void ngp_calc_gradient(
    $uint &batch_size, 
    int input_dim, 
    $buffer<half4> &input, 
    $buffer<half4> &output_grad, 
    $buffer<half2> &feature_grad, 
    $buffer<int> &level_offsets
) {
    set_block_size(256, 1);
    $uint tid = $dispatch_x;
    $uint level = $dispatch_y;
    $uint level_offset = level_offsets.read(level);
    $uint level_size = level_offsets.read(level + 1) - level_offset;

    $uint grid_res = (1u << level) * N_min;
    $bool use_hash = (pow(grid_res.cast<float>(), (float)input_dim) > level_size + 50) | (powi(grid_res, input_dim) > level_size);

    $float4 in[4];
    $float4 pos[4];
    $uint4 grid_idx[4];

    for (int i = 0; i < input_dim; i++) {
        in[i] = input.read(tid + i*batch_size/4);
    }
    trans_input(in, input_dim);
    for (int i = 0; i < 4; i++) {
        calc_pos(in[i], pos[i], grid_idx[i], grid_res);
    }

    $half4 g_out[2];
    g_out[0] = output_grad.read(tid + level*2*batch_size/4);
    g_out[1] = output_grad.read(tid + (level*2+1)*batch_size/4);

    $float w;
    $float2 g;
    $half2 tg;
    $uint4 idx;
    for (int i = 0; i < 4; i++) {
        g[0] = g_out[0][i];
        g[1] = g_out[1][i];
        for (int t = 0; t < powi(2, input_dim); t++) {
            w = 1;
            for (int d = 0; d < input_dim; d++) {
                if ((t & (1 << d)) == 0) {
                    w *= 1 - pos[i][d];
                    idx[d] = grid_idx[i][d];
                }
                else {
                    w *= pos[i][d];
                    idx[d] = grid_idx[i][d] + 1;
                }
            }
            tg = (w*batch_size)*g;
            feature_grad.write(level_offset + table_idx(input_dim, idx, grid_res, level_size, use_hash), tg);
        }
    }
}

Shader2D<uint, Buffer<half4>, Buffer<half4>, Buffer<half2>, Buffer<int>> ngp_calc_gradient_shader[5];
Shader1D<Buffer<half2>> clear_grad_shader;

void NGPEncodeLayer::backward(
    const BufferView<half4> fwd_input,
    const BufferView<half4> fwd_output,
    BufferView<half4> output_grad,
    BufferView<half4> input_grad,
    BufferView<half4> arena
) {
    if (!ngp_calc_gradient_shader[input_dim()]) {
        Kernel2D ngp_calc_gradient_kernel = [&](
            $uint batch_size,
            $buffer<half4> input, 
            $buffer<half4> output_grad, 
            $buffer<half2> feature_grad, 
            $buffer<int> level_offsets
        ) {
            ngp_calc_gradient(batch_size, input_dim(), input, output_grad, feature_grad, level_offsets);
        };
        ngp_calc_gradient_shader[input_dim()] = global::device().compile(ngp_calc_gradient_kernel);
    }
    if (!clear_grad_shader) {
        Kernel1D clear_grad_kernel = []($buffer<half2> feature_grad) {
            set_block_size(256);
            $uint tid = $dispatch_x;
            feature_grad.write(tid, $half2(0, 0));
        };
        clear_grad_shader = global::device().compile(clear_grad_kernel);
    }
    const uint batch_size = output_grad.size()*4 / output_dim();
    global::cmd_list()
        << clear_grad_shader(_feature_gradient).dispatch(_feature_gradient.size())
        << ngp_calc_gradient_shader[input_dim()](batch_size, fwd_input, output_grad, _feature_gradient, level_offset_buffer).dispatch(batch_size/4, levels);
}

void NGPEncodeLayer::optimize() {
    optim.optimize(_feature_table.view().as<half4>(), _feature_gradient.view().as<half4>());
}