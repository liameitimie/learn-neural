#include <luisa/dsl/sugar.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>
#include <linear_layer.h>
#include <global.h>
#include <activation_func.h>
#include <gpu_rands.h>

using namespace luisa;
using namespace luisa::compute;

float weight_scale_siren(int input_dim) {
    return sqrt(6.0f / (float)input_dim);
}
float weight_scale_xavier(int input_dim, int output_dim) {
    return sqrt(6.0f / (float)(input_dim + output_dim));
}

const uint tile_k = 64;
namespace matmuls {
    void act_mm_crr_32_x_32(activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d);
    void act_mm_rrr_32_x_32(activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d);
    void act_mm_bias_crr_32_x_32(activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> c, BufferView<half4> d);
    // void act_mm_bias_rrr_32_x_32(activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> c, BufferView<half4> d);

    void act_mm_crr_l4_x_32(int m, activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d);
    void act_mm_rrr_l4_x_32(int m, activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d);
    void act_mm_bias_crr_l4_x_32(int m, activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> c, BufferView<half4> d);

    void act_mm_crr_32_x_l4(int n, activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d);
    void act_mm_rrr_32_x_l4(int n, activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d);
    void act_mm_bias_crr_32_x_l4(int n, activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> c, BufferView<half4> d);

    void mm_rcc_32_32_x(uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d, BufferView<half4> splitk_tmp);
    void mm_rcc_l4_32_x(int m, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d, BufferView<half4> splitk_tmp);
    void mm_rcc_32_l4_x(int n, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d, BufferView<half4> splitk_tmp);

    int mm_fid1(int m, int n, int k) {
        if (m == 32 && k == 32) return 0;
        if (m == 32 || k == 32) {
            if (m <= 4) return 1;
            if (k <= 4) return 2;
        }
        return -1;
    }
    void act_mm_crr(int m, int n, int k, activation::Activation act, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d) {
        int fid = mm_fid1(m, n, k);
        switch (fid) {
            case 0: act_mm_crr_32_x_32(act, n, a, b, d); break;
            case 1: act_mm_crr_l4_x_32(m, act, n, a, b, d); break;
            case 2: act_mm_crr_32_x_l4(k, act, n, a, b, d); break;
            default: {
                fmt::print("unimpl act_mm_crr for {}x{}x{}\n", m, n, k);
                exit(0);
            }
        }
    }
    void act_mm_bias_crr(int m, int n, int k, activation::Activation act, BufferView<half4> a, BufferView<half4> b, BufferView<half4> c, BufferView<half4> d) {
        int fid = mm_fid1(m, n, k);
        switch (fid) {
            case 0: act_mm_bias_crr_32_x_32(act, n, a, b, c, d); break;
            case 1: act_mm_bias_crr_l4_x_32(m, act, n, a, b, c, d); break;
            case 2: act_mm_bias_crr_32_x_l4(k, act, n, a, b, c, d); break;
            default: {
                fmt::print("unimpl act_mm_bias_crr for {}x{}x{}\n", m, n, k);
                exit(0);
            }
        }
    }
    void act_mm_rrr(int m, int n, int k, activation::Activation act, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d) {
        int fid = mm_fid1(m, n, k);
        switch (fid) {
            case 0: act_mm_rrr_32_x_32(act, n, a, b, d); break;
            case 1: act_mm_rrr_l4_x_32(m, act, n, a, b, d); break;
            case 2: act_mm_rrr_32_x_l4(k, act, n, a, b, d); break;
            default: {
                fmt::print("unimpl act_mm_rrr for {}x{}x{}\n", m, n, k);
                exit(0);
            }
        }
    }
    int mm_fid2(int m, int n, int k) {
        if (m == 32 && n == 32) return 0;
        if (m == 32 || n == 32) {
            if (m <= 4) return 1;
            if (n <= 4) return 2;
        }
        return -1;
    }
    void mm_rcc_splitk(int m, int n, int k, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d, BufferView<half4> splitk_tmp) {
        int fid = mm_fid2(m, n, k);
        switch (fid) {
            case 0: mm_rcc_32_32_x(k, a, b, d, splitk_tmp); break;
            case 1: mm_rcc_l4_32_x(m, k, a, b, d, splitk_tmp); break;
            case 2: mm_rcc_32_l4_x(n, k, a, b, d, splitk_tmp); break;
            default: {
                fmt::print("unimpl mm_rcc_splitk for {}x{}x{}\n", m, n, k);
                exit(0);
            }
        }
    }
}

void LinearLayer::forward(const BufferView<half4> input, BufferView<half4> output) {
    const uint batch_size = input.size()*4 / input_dim();
    if (use_bias) {
        matmuls::act_mm_bias_crr(output_dim(), batch_size, input_dim(), act, _weight, input, _bias, output);
    }
    else {
        matmuls::act_mm_crr(output_dim(), batch_size, input_dim(), act, _weight, input, output);
    }
}

Shader2D<uint, Buffer<half4>, Buffer<half4>> reduce_shader1;
Shader1D<uint, uint, Buffer<half4>, Buffer<half4>> reduce_shader2;

void reduce(int len, int dim, BufferView<half4> in, BufferView<half4> out, BufferView<half4> arena) {
    if (!reduce_shader1) {
        Kernel2D reduce1 = []($uint len, $buffer<half4> in, $buffer<half4> arena) {
            set_block_size(256, 1);
            $shared<float4> smem{256};
            $uint tid = $dispatch_x % 256;
            $uint bid = $dispatch_x / 256;
            $uint dim = $dispatch_y;

            $float4 t1;
            $float4 t2;
            $if (tid + bid*512 < len/4) { t1 = in.read(tid + bid*512 + dim*len/4); };
            $if (tid + 256 + bid*512 < len/4) { t2 = in.read(tid + 256 + bid*512 + dim*len/4); };
            smem[tid] = t1 + t2;
            sync_block();

            for (int s = 128; s > 0; s >>= 1) {
                $if (tid < s) {
                    smem[tid] += smem[tid + s];
                };
                sync_block();
            }
            $if (tid == 0) {
                $uint s = (len/4 + 511) / 512;
                $half4 t = smem[0];
                arena.write(bid + dim*s, t);
            };
        };
        reduce_shader1 = global::device().compile(reduce1);
    }
    if (!reduce_shader2) {
        Kernel1D reduce2 = []($uint len, $uint dim, $buffer<half4> arena, $buffer<half4> out) {
            // 每4维度一个block，为了适应half4的buffer读写
            set_block_size(256);
            $shared<float4> smem{256};
            $uint step = (len/4 + 511) / 512;
            $uint tid = $dispatch_x % 256;
            $uint bid = $dispatch_x / 256;

            $uint did = tid / 64;
            $uint sid = tid % 64;

            $float4 t, t1;
            $if (did + bid*4 < dim) {
                $for (i, sid, step, 64u) {
                    t1 = arena.read(i + (did + bid*4)*step);
                    t += t1;
                };
                smem[tid] = t;
            };
            sync_block();
            for (int s = 32; s > 0; s >>= 1) {
                $if (sid < s & did + bid*4 < dim) {
                    smem[tid] += smem[tid + s];
                };
                sync_block();
            }
            $if (sid == 0) {
                t = smem[tid];
                smem[0][did] = t.x + t.y + t.z + t.w; 
            };
            sync_block();
            $if (tid == 0) {
                $half4 t = smem[0];
                out.write(bid, t);
            };
        };
        reduce_shader2 = global::device().compile(reduce2);
    }
    int step = (len/4 + 511) / 512;
    global::cmd_list() << reduce_shader1(len, in, arena).dispatch(step*256, dim)
        << reduce_shader2(len, dim, arena, out).dispatch((dim+3)/4 * 256);
}

Shader1D<Buffer<half4>, Buffer<half4>, Buffer<half4>> act_backward_shader[activation::NUM];

void LinearLayer::backward(
    const BufferView<half4> fwd_input,
    const BufferView<half4> fwd_output,
    BufferView<half4> output_grad,
    BufferView<half4> input_grad,
    BufferView<half4> arena
) {
    const uint batch_size = output_grad.size()*4 / output_dim();
    if (act != activation::None) {
        if (!act_backward_shader[act]) {
            Kernel1D act_backward = [&]($buffer<half4> g_out, $buffer<half4> fwd_out, $buffer<half4> store) {
                set_block_size(256);
                $uint tid = $dispatch_x;
                $half4 g = g_out.read(tid);
                $half4 f = fwd_out.read(tid);
                for (int i = 0; i < 4; i++) {
                    activation::backward(act, g[i], f[i]);
                }
                store.write(tid, g);
            };
            act_backward_shader[act] = global::device().compile(act_backward);
        }
        global::cmd_list() << act_backward_shader[act](output_grad, fwd_output, output_grad).dispatch(output_grad.size());
    }
    if (input_grad) {
        matmuls::act_mm_rrr(input_dim(), batch_size, output_dim(), activation::None, _weight, output_grad, input_grad);
    }
    
    if (!arena) {
        fmt::print("error: linear layer backward need a arena buffer\n");
        exit(0);
    }
    matmuls::mm_rcc_splitk(output_dim(), input_dim(), batch_size, output_grad, fwd_input, _weight_grad, arena.subview(0, gradweight_arena_size(batch_size)/4));
    if (use_bias) {
        reduce(batch_size, output_dim(), output_grad, _bias_grad, arena.subview(gradweight_arena_size(batch_size)/4, gradbias_arena_size(batch_size)/4));
    }
}

void LinearLayer::optimize() {
    optim.optimize(param_buffer, grad_buffer);
}

int LinearLayer::gradweight_arena_size(int batch_size) {
    int pad_output_dim = (output_dim()+3)/4*4;
    return input_dim() * pad_output_dim * (batch_size / tile_k);
}
int LinearLayer::gradbias_arena_size(int batch_size) {
    int size = 0;
    if (use_bias) {
        int step = (batch_size/4 + 511) / 512;
        size += output_dim() * step * 4;
    }
    return size;
}

int LinearLayer::arena_size(int batch_size) {
    return gradweight_arena_size(batch_size) + gradbias_arena_size(batch_size);
}

Shader1D<Buffer<half4>, float> reset_weight_shader;
Shader1D<Buffer<half4>, float> reset_bias_shader;

void LinearLayer::reset_parameters() {
    if (!reset_weight_shader) {
        Kernel1D reset_weight = []($buffer<half4> weight, $float w_scale) {
            $uint tid = $dispatch_x;
            $half4 w;
            $for (i, 4) {
                $uint s = tea(tid + 233, i + 114514).x;
                $float f = as_uniform(s);
                w[i] = (f * 2 - 1) * w_scale;
            };
            weight.write(tid, w);
        };
        reset_weight_shader = global::device().compile(reset_weight);
    }
    global::cmd_list() << reset_weight_shader(_weight, weight_scale).dispatch(_weight.size());

    if (use_bias) {
        if (!reset_bias_shader) {
            Kernel1D reset_bias = []($buffer<half4> bias, $float b_scale) {
                $uint tid = $dispatch_x;
                $half4 b;
                $for (i, 4) {
                    $uint s = tea(tid + 114514, i + 233).x;
                    $float f = as_uniform(s);
                    b[i] = (f * 2 - 1) * b_scale;
                };
                bias.write(tid, b);
            };
            reset_bias_shader = global::device().compile(reset_bias);
        }

        float b_scale = 1 / sqrt(input_dim());
        global::cmd_list() << reset_bias_shader(_bias, b_scale).dispatch(_bias.size());
    }
}

LinearLayer::LinearLayer(int input_dim, int output_dim, bool use_bias, activation::Activation act, float w_scale, AdamConfig optim_cfg):
    DiffLayer(input_dim, output_dim),
    use_bias(use_bias),
    act(act),
    weight_scale(w_scale),
    optim(optim_cfg)
{
    if (!((input_dim == 32 && output_dim == 32) || (output_dim == 32 && input_dim <= 4) || (input_dim == 32 && output_dim <= 4))) {
        LUISA_ERROR_WITH_LOCATION(
            "unimpl linear layer with input_dim={}, output_dim={}\n",
        input_dim, output_dim);
    }
    // for buffer size
    if (output_dim < 4) {
        output_dim = 4;
    }

    const int weight_size = input_dim * output_dim;
    const int bias_size = output_dim;
    int param_size = weight_size;
    if (use_bias) param_size += bias_size;

    param_buffer = global::device().create_buffer<half4>(param_size / 4);
    grad_buffer = global::device().create_buffer<half4>(param_size / 4);

    _weight = param_buffer.view(0, weight_size / 4);
    _weight_grad = grad_buffer.view(0, weight_size / 4);

    if (use_bias) {
        _bias = param_buffer.view(weight_size / 4, bias_size / 4);
        _bias_grad = grad_buffer.view(weight_size / 4, bias_size / 4);
    }

    reset_parameters();
    optim.init(param_buffer);
}


/////////////////
// matmul impl //
/////////////////

void load_32_32($buffer<half4> &buf, $shared<half4> &smem, $uint tid) {
    // $uint tid = $dispatch_x % 256;
    smem[tid] = buf.read(tid);
}
void load_trans_32_32($buffer<half4> &buf, $shared<half4> &smem, $uint tid) {
    // $uint tid = $dispatch_x % 256;
    $half4 tmp = buf.read(tid);
    // for (int i = 0; i < 4; i++) {
    $for (i, 4) {
        smem[tid/32 + (tid%8*4 + i)*8][tid/8%4] = tmp[i];
    };
}
void load_32_l4(int x, $buffer<half4> &buf, $shared<half4> &smem, $uint tid) {
    assert(x <= 4);
    // $uint tid = $dispatch_x % 256;
    $if (tid < x*8) {
        smem[tid] = buf.read(tid);
    };
}
void load_trans_l4_32(int x, $buffer<half4> &buf, $shared<half4> &smem, $uint tid) {
    assert(x <= 4);
    // $uint tid = $dispatch_x % 256;
    $if (tid < x*8) {
        $half4 tmp = buf.read(tid);
        for (int i = 0; i < 4; i++) {
            smem[tid%8*4 + i][tid/8] = tmp[i];
        }
    };
}

// assert buffer is padding with 4
void load_l4_32(int x, $buffer<half4> &buf, $shared<half4> &smem, $uint tid) {
    assert(x <= 4);
    // $uint tid = $dispatch_x % 256;
    $if (tid < 32) {
        smem[tid] = buf.read(tid);
    };
}
// assert buffer is padding with 4
void load_trans_32_l4(int x, $buffer<half4> &buf, $shared<half4> &smem, $uint tid) {
    assert(x <= 4);
    // $uint tid = $dispatch_x % 256;
    $if (tid < 32) {
        $half4 tmp = buf.read(tid);
        for (int i = 0; i < x; i++) {
            smem[tid/4 + i*8][tid%4] = tmp[i];
        }
    };
}

void load_512_l4(int x, $buffer<half4> &buf, $shared<half4> &smem, $uint offset, $uint stride, $uint tid) {
    assert(x <= 4);
    // $uint tid = $dispatch_x % 256;
    if (x == 1) {
        $if (tid < 128) {
            smem[tid] = buf.read(tid + offset/4);
        };
    } else {
        smem[tid] = buf.read(tid%128 + offset/4 + tid/128*stride/4);
        if (x > 2) {
            if (x == 3) {
                $if (tid < 128) {
                    smem[tid + 256] = buf.read(tid + offset/4 + 2*stride/4);
                };
            } else {
                smem[tid + 256] = buf.read(tid%128 + offset/4 + (2 + tid/128)*stride/4);
            }
        }
    }
}

void out_product_l4_l4_r(int x, int y, $half4 &a, $half4 &b, $half4 *acc) {
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            acc[i][j] += a[i] * b[j];
        }
    }
}
void out_product_l4_l4_c(int x, int y, $half4 &a, $half4 &b, $half4 *acc) {
    for (int j = 0; j < y; j++) {
        for (int i = 0; i < x; i++) {
            acc[j][i] += a[i] * b[j];
        }
    }
}

// void out_product_8_8_r($half4 *a, $half4 *b, $half4 *acc) {
void out_product_8_8_r($array<half4, 2> &a, $array<half4, 2> &b, $array<half4, 16> &acc) {
    for (int tx = 0; tx < 2; tx++) {
        for (int ty = 0; ty < 2; ty++) {
    // $for (tx, 2) {
    //     $for (ty, 2) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
            // $for (i, 4) {
            //     $for (j, 4) {
                    acc[i + (ty + tx*2)*4][j] += a[tx][i] * b[ty][j];
                };
            };
        };
    };
}

// 放弃挣扎
// void store_res_tile($buffer<half4> &buf, $shared<half4> &smem, $half4 *acc, $uint stride) {
void store_res_tile($buffer<half4> &buf, $shared<half4> &smem, $array<half4, 16> &acc, $uint stride, $uint tid, $uint bid) {
    // $uint tid = $dispatch_x % 256;
    // $uint bid = $dispatch_x / 256;
    $uint x_ofs = tid%32/8;
    $uint y_ofs = tid%8 + tid/32*16;
    // for (int t = 0; t < 2; t++) {
    //     for (int i = 0; i < 4; i++) {
    $for (t, 2) {
        $for (i, 4) {
            sync_block();
            smem[y_ofs + x_ofs*128] = acc[i + t*2*4];
            smem[y_ofs + 8 + x_ofs*128] = acc[i + (t*2 + 1)*4];
            sync_block();

            buf.write(tid%128 + bid*128 + (i + t*16 + tid/128*4)*stride/4, smem[tid]);
            buf.write(tid%128 + bid*128 + (i + t*16 + (tid/128 + 2)*4)*stride/4, smem[tid + 256]);
        };
    };
}


void mm_32_x_32_impl(
    $uint &x, 
    $buffer<half4> &a, bool trans_a,
    $buffer<half4> &b, 
    $buffer<half4> *c, 
    $buffer<half4> &d,
    activation::Activation act
) {
    set_block_size(256);
    $shared<half4> a_tile{256};
    $shared<half4> b_tile{512};
    $array<half4, 16> acc;
    $array<half4, 2> a_frag;
    $array<half4, 2> b_frag;
    // $half4 acc[16];
    // $half4 a_frag[2];
    // $half4 b_frag[2];

    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;

    $uint x_ofs = tid%32/8;
    $uint y_ofs = tid%8 + tid/32*16;

    if (!trans_a) load_32_32(a, a_tile, tid);
    else load_trans_32_32(a, a_tile, tid);

    $for (k, 0, 32, 4) {
        sync_block();
        load_512_l4(4, b, b_tile, bid*512+k*x, x, tid);
        sync_block();
        $for (k1, 4) {
            for (int t = 0; t < 2; t++) {
                a_frag[t] = a_tile[x_ofs + t*4 + (k + k1)*8];
                b_frag[t] = b_tile[y_ofs + t*8 + k1*128];
            }
            out_product_8_8_r(a_frag, b_frag, acc);
        };
    };
    if (c != nullptr) {
        // apply bias
        $if (tid < 8) {
            b_tile[tid] = c->read(tid);
        };
        sync_block();
        b_frag[0] = b_tile[x_ofs];
        b_frag[1] = b_tile[x_ofs + 4];
        $for (i, 16) {
            acc[i] += make_half4(b_frag[i/8][i%4]);
        };
    }
    if (act != activation::None) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 4; j++) {
        // $for (i, 16) {
        //     $for (j, 4) {
                activation::forward(act, acc[i][j]);
            };
        };
    }
    store_res_tile(d, b_tile, acc, x, tid, bid);
}

void mm_l4_x_32_impl(
    int m,
    $uint &x, 
    $buffer<half4> &a, bool trans_a,
    $buffer<half4> &b, 
    $buffer<half4> *c, 
    $buffer<half4> &d,
    activation::Activation act
) {
    set_block_size(256);
    $shared<half4> a_tile{32};
    $half4 acc[4];
    $half4 a_frag;
    $half4 b_frag;

    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;

    // assert a is padding with 4
    if (!trans_a) load_l4_32(m, a, a_tile, tid);
    else load_trans_l4_32(m, a, a_tile, tid);
    sync_block();

    $for (k, 32) {
        a_frag = a_tile[k];
        for (int i = m; i < 4; i++) a_frag[i] = 0;
        b_frag = b.read(tid + bid*256 + k*x/4);
        out_product_l4_l4_r(m, 4, a_frag, b_frag, acc);
    };
    if (c != nullptr) {
        // apply bias
        $if (tid == 0) {
            a_tile[tid] = c->read(tid);
        };
        sync_block();
        a_frag = a_tile[0];
        for (int i = 0; i < m; i++) {
            acc[i] += make_half4(a_frag[i]);
        }
    }
    if (act != activation::None) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < 4; j++) {
                activation::forward(act, acc[i][j]);
            }
        }
    }
    for (int i = 0; i < m; i++) {
        d.write(tid + bid*256 + i*x/4, acc[i]);
    }
}

void mm_32_x_l4_impl(
    int n,
    $uint &x, 
    $buffer<half4> &a, bool trans_a,
    $buffer<half4> &b, 
    $buffer<half4> *c, 
    $buffer<half4> &d,
    activation::Activation act
) {
    set_block_size(256);
    $shared<half4> a_tile{32};
    $shared<half4> b_tile{512};
    $array<half4, 16> acc;
    $array<half4, 2> a_frag;
    $array<half4, 2> b_frag;

    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;

    $uint x_ofs = tid%32/8;
    $uint y_ofs = tid%8 + tid/32*16;

    if (!trans_a) load_32_l4(n, a, a_tile, tid);
    else load_trans_32_l4(n, a, a_tile, tid);
    load_512_l4(n, b, b_tile, bid*512, x, tid);
    sync_block();

    $for (k, n) {
        for (int t = 0; t < 2; t++) {
            a_frag[t] = a_tile[x_ofs + t*4 + k*8];
            b_frag[t] = b_tile[y_ofs + t*8 + k*128];
        }
        out_product_8_8_r(a_frag, b_frag, acc);
    };

    if (c != nullptr) {
        // apply bias
        $if (tid < 8) {
            b_tile[tid] = c->read(tid);
        };
        sync_block();
        b_frag[0] = b_tile[x_ofs];
        b_frag[1] = b_tile[x_ofs + 4];
        $for (i, 16) {
            acc[i] += make_half4(b_frag[i/8][i%4]);
        };
    }
    if (act != activation::None) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 4; j++) {
        // $for (i, 16) {
        //     $for (j, 4) {
                activation::forward(act, acc[i][j]);
            };
        };
    }
    store_res_tile(d, b_tile, acc, x, tid, bid);
}

void load_trans_32_4($buffer<half4> &buf, $shared<half4> &smem, $uint offset, $uint stride, $uint tid) {
    $half4 tmp = buf.read(offset/4 + tid*stride/4);
    for (int i = 0; i < 4; i++) {
        smem[tid/4 + i*8][tid%4] = tmp[i];
    }
}
void load_trans_l4_4(int x, $buffer<half4> &buf, $shared<half4> &smem, $uint offset, $uint stride, $uint tid) {
    $if (tid < x) {
        $half4 tmp = buf.read(offset/4 + tid*stride/4);
        for (int i = 0; i < 4; i++) {
            smem[i][tid] = tmp[i];
        }
    };
}

void mm_32_32_x_impl(
    $uint &x, 
    $buffer<half4> &a,
    $buffer<half4> &b,
    $buffer<half4> &splitk_tmp
) {
    set_block_size(32);
    $shared<half4> a_tile{32};
    $shared<half4> b_tile{32};
    $half4 acc[8];
    $half4 a_frag;
    $half4 b_frag[2];

    $uint tid = $dispatch_x % 32;
    $uint bid = $dispatch_x / 32;
    $uint x_ofs = tid % 8;
    $uint y_ofs = tid / 8;

    $for (k, tile_k/4) {
        load_trans_32_4(a, a_tile, k*4 + bid*tile_k, x, tid);
        load_trans_32_4(b, b_tile, k*4 + bid*tile_k, x, tid);
        $for (k1, 4) {
            a_frag = a_tile[x_ofs + k1*8];
            b_frag[0] = b_tile[y_ofs + k1*8];
            b_frag[1] = b_tile[y_ofs + 4 + k1*8];
            out_product_l4_l4_c(4, 4, a_frag, b_frag[0], acc);
            out_product_l4_l4_c(4, 4, a_frag, b_frag[1], acc + 4);
        };
    };
    for (int i = 0; i < 4; i++) {
        splitk_tmp.write(x_ofs + (y_ofs*4+i)*8 + bid*(32*32/4), acc[i]);
    }
    for (int i = 0; i < 4; i++) {
        splitk_tmp.write(x_ofs + ((y_ofs+4)*4+i)*8 + bid*(32*32/4), acc[i+4]);
    }
}

void mm_l4_32_x_impl(
    int m,
    $uint &x, 
    $buffer<half4> &a,
    $buffer<half4> &b,
    $buffer<half4> &splitk_tmp
) {
    set_block_size(32);
    $shared<half4> a_tile{32};
    $shared<half4> b_tile{32};
    $half4 acc[4];
    $half4 a_frag;
    $half4 b_frag;
    $uint tid = $dispatch_x % 32;
    $uint bid = $dispatch_x / 32;
    $for (k, tile_k/4) {
        load_trans_l4_4(m, a, a_tile, k*4 + bid*tile_k, x, tid);
        load_trans_32_4(b, b_tile, k*4 + bid*tile_k, x, tid);
        sync_block();
        a_frag = a_tile[tid/8];
        b_frag = b_tile[tid];
        out_product_l4_l4_c(m, 4, a_frag, b_frag, acc);
    };
    for (int i = 0; i < 4; i++) {
        a_tile[tid] = acc[i];
        sync_block();
        for (int s = 16; s >= 8; s >>= 1) {
            $if (tid < s) {
                a_tile[tid] += a_tile[tid + s];
            };
            sync_block();
        }
        $if (tid < 8) {
            b_tile[tid*4 + i] = a_tile[tid];
        };
    }
    splitk_tmp.write(tid + bid*32, b_tile[tid]);
}

void mm_32_l4_x_impl(
    int n,
    $uint &x, 
    $buffer<half4> &a,
    $buffer<half4> &b,
    $buffer<half4> &splitk_tmp
) {
    set_block_size(32);
    $shared<half4> a_tile{32};
    $shared<half4> b_tile{32};
    $half4 acc[4];
    $half4 a_frag;
    $half4 b_frag;
    $uint tid = $dispatch_x % 32;
    $uint bid = $dispatch_x / 32;
    $for (k, tile_k/4) {
        load_trans_32_4(a, a_tile, k*4 + bid*tile_k, x, tid);
        load_trans_l4_4(n, b, b_tile, k*4 + bid*tile_k, x, tid);
        sync_block();
        a_frag = a_tile[tid];
        b_frag = b_tile[tid/8];
        out_product_l4_l4_c(4, n, a_frag, b_frag, acc);
    };
    for (int i = 0; i < n; i++) {
        a_tile[tid] = acc[i];
        sync_block();
        for (int s = 16; s >= 8; s >>= 1) {
            $if (tid < s) {
                a_tile[tid] += a_tile[tid + s];
            };
            sync_block();
        }
        $if (tid < 8) {
            b_tile[tid + i*8] = a_tile[tid];
        };
        sync_block();
    }
    $if (tid < 8*n) {
        splitk_tmp.write(tid + bid*(8*n), b_tile[tid]);
    };
}

void splitk_reduce_impl($uint &len, $uint &stride, $buffer<half4> &splitk_tmp, $buffer<half4> &store) {
    $uint tid = $dispatch_x;
    $half4 acc;
    $for (t, len / tile_k) {
        acc += splitk_tmp.read(tid + t*stride);
    };
    // for (int i = 0; i < 4; i++) {
    //     acc[i] = acc[i] / 16384;
    // }
    store.write(tid, acc);
}

namespace matmuls {
    using mm_shader_t = Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>>;
    using mm_bias_shader_t = Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>, Buffer<half4>>;
    // using mm_splitk_shader_t = Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>, Buffer<half4>>;

    mm_shader_t act_mm_crr_32_x_32_shader[activation::NUM];
    mm_shader_t act_mm_rrr_32_x_32_shader[activation::NUM];
    mm_bias_shader_t act_mm_bias_crr_32_x_32_shader[activation::NUM];
    // mm_bias_shader_t act_mm_bias_rrr_32_x_32_shader[activation::NUM];

    mm_shader_t act_mm_crr_l4_x_32_shader[5][activation::NUM];
    mm_shader_t act_mm_rrr_l4_x_32_shader[5][activation::NUM];
    mm_bias_shader_t act_mm_bias_crr_l4_x_32_shader[5][activation::NUM];

    mm_shader_t act_mm_crr_32_x_l4_shader[5][activation::NUM];
    mm_shader_t act_mm_rrr_32_x_l4_shader[5][activation::NUM];
    mm_bias_shader_t act_mm_bias_crr_32_x_l4_shader[5][activation::NUM];

    mm_shader_t mm_rcc_32_32_x_shader;
    mm_shader_t mm_rcc_l4_32_x_shader[5];
    mm_shader_t mm_rcc_32_l4_x_shader[5];
    Shader1D<uint, uint, Buffer<half4>, Buffer<half4>> splitk_reduce_shader;

    void act_mm_crr_32_x_32(activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d) {
        if (!act_mm_crr_32_x_32_shader[act]) {
            Kernel1D act_mm_crr_32_x_32 = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> d) {
                mm_32_x_32_impl(x, a, false, b, nullptr, d, act);
            };
            act_mm_crr_32_x_32_shader[act] = global::device().compile(act_mm_crr_32_x_32);
        }
        global::cmd_list() << act_mm_crr_32_x_32_shader[act](x, a, b, d).dispatch(x / 2);
    }

    void act_mm_rrr_32_x_32(activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d) {
        if (!act_mm_rrr_32_x_32_shader[act]) {
            Kernel1D act_mm_rrr_32_x_32 = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> d) {
                mm_32_x_32_impl(x, a, true, b, nullptr, d, act);
            };
            act_mm_rrr_32_x_32_shader[act] = global::device().compile(act_mm_rrr_32_x_32);
        }
        global::cmd_list() << act_mm_rrr_32_x_32_shader[act](x, a, b, d).dispatch(x / 2);
    }

    void act_mm_bias_crr_32_x_32(activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> c, BufferView<half4> d) {
        if (!act_mm_bias_crr_32_x_32_shader[act]) {
            Kernel1D act_mm_bias_crr_32_x_32 = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> c, $buffer<half4> d) {
                mm_32_x_32_impl(x, a, false, b, &c, d, act);
            };
            act_mm_bias_crr_32_x_32_shader[act] = global::device().compile(act_mm_bias_crr_32_x_32);
        }
        global::cmd_list() << act_mm_bias_crr_32_x_32_shader[act](x, a, b, c, d).dispatch(x / 2);
    }

    void act_mm_crr_l4_x_32(int m, activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d) {
        if (!act_mm_crr_l4_x_32_shader[m][act]) {
            Kernel1D act_mm_crr_l4_x_32 = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> d) {
                mm_l4_x_32_impl(m, x, a, false, b, nullptr, d, act);
            };
            act_mm_crr_l4_x_32_shader[m][act] = global::device().compile(act_mm_crr_l4_x_32);
        }
        global::cmd_list() << act_mm_crr_l4_x_32_shader[m][act](x, a, b, d).dispatch(x / 4);
    }

    void act_mm_rrr_l4_x_32(int m, activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d) {
        if (!act_mm_rrr_l4_x_32_shader[m][act]) {
            Kernel1D act_mm_rrr_l4_x_32 = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> d) {
                mm_l4_x_32_impl(m, x, a, true, b, nullptr, d, act);
            };
            act_mm_rrr_l4_x_32_shader[m][act] = global::device().compile(act_mm_rrr_l4_x_32);
        }
        global::cmd_list() << act_mm_rrr_l4_x_32_shader[m][act](x, a, b, d).dispatch(x / 4);
    }

    void act_mm_bias_crr_l4_x_32(int m, activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> c, BufferView<half4> d) {
        if (!act_mm_bias_crr_l4_x_32_shader[m][act]) {
            Kernel1D act_mm_bias_crr_l4_x_32 = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> c, $buffer<half4> d) {
                mm_l4_x_32_impl(m, x, a, false, b, &c, d, act);
            };
            act_mm_bias_crr_l4_x_32_shader[m][act] = global::device().compile(act_mm_bias_crr_l4_x_32);
        }
        global::cmd_list() << act_mm_bias_crr_l4_x_32_shader[m][act](x, a, b, c, d).dispatch(x / 4);
    }

    void act_mm_crr_32_x_l4(int n, activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d) {
        if (!act_mm_crr_32_x_l4_shader[n][act]) {
            Kernel1D act_mm_crr_32_x_l4 = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> d) {
                mm_32_x_l4_impl(n, x, a, false, b, nullptr, d, act);
            };
            act_mm_crr_32_x_l4_shader[n][act] = global::device().compile(act_mm_crr_32_x_l4);
        }
        global::cmd_list() << act_mm_crr_32_x_l4_shader[n][act](x, a, b, d).dispatch(x / 2);
    }

    void act_mm_rrr_32_x_l4(int n, activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d) {
        if (!act_mm_rrr_32_x_l4_shader[n][act]) {
            Kernel1D act_mm_rrr_32_x_l4 = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> d) {
                mm_32_x_l4_impl(n, x, a, true, b, nullptr, d, act);
            };
            act_mm_rrr_32_x_l4_shader[n][act] = global::device().compile(act_mm_rrr_32_x_l4);
        }
        global::cmd_list() << act_mm_rrr_32_x_l4_shader[n][act](x, a, b, d).dispatch(x / 2);
    }

    void act_mm_bias_crr_32_x_l4(int m, activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> c, BufferView<half4> d) {
        if (!act_mm_bias_crr_32_x_l4_shader[m][act]) {
            Kernel1D act_mm_bias_crr_32_x_l4 = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> c, $buffer<half4> d) {
                mm_32_x_l4_impl(m, x, a, false, b, &c, d, act);
            };
            act_mm_bias_crr_32_x_l4_shader[m][act] = global::device().compile(act_mm_bias_crr_32_x_l4);
        }
        global::cmd_list() << act_mm_bias_crr_32_x_l4_shader[m][act](x, a, b, c, d).dispatch(x / 2);
    }

    // void act_mm_bias_rrr_32_x_32(activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> c, BufferView<half4> d);

    void mm_rcc_32_32_x(uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d, BufferView<half4> splitk_tmp) {
        if (!mm_rcc_32_32_x_shader) {
            Kernel1D mm_rcc_32_32_x = []($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> splitk_tmp) {
                mm_32_32_x_impl(x, a, b, splitk_tmp);
            };
            mm_rcc_32_32_x_shader = global::device().compile(mm_rcc_32_32_x);
        }
        if (!splitk_reduce_shader) {
            Kernel1D splitk_reduce = []($uint x, $uint stride, $buffer<half4> splitk_tmp, $buffer<half4> store) {
                splitk_reduce_impl(x, stride, splitk_tmp, store);
            };
            splitk_reduce_shader = global::device().compile(splitk_reduce);
        }
        global::cmd_list()
            << mm_rcc_32_32_x_shader(x, a, b, splitk_tmp).dispatch(x / (tile_k / 32))
            << splitk_reduce_shader(x, 32*32/4, splitk_tmp, d).dispatch(32*32/4);
    }

    void mm_rcc_l4_32_x(int m, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d, BufferView<half4> splitk_tmp) {
        if (!mm_rcc_l4_32_x_shader[m]) {
            Kernel1D mm_rcc_l4_32_x = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> splitk_tmp) {
                mm_l4_32_x_impl(m, x, a, b, splitk_tmp);
            };
            mm_rcc_l4_32_x_shader[m] = global::device().compile(mm_rcc_l4_32_x);
        }
        if (!splitk_reduce_shader) {
            Kernel1D splitk_reduce = []($uint x, $uint stride, $buffer<half4> splitk_tmp, $buffer<half4> store) {
                splitk_reduce_impl(x, stride, splitk_tmp, store);
            };
            splitk_reduce_shader = global::device().compile(splitk_reduce);
        }
        global::cmd_list()
            << mm_rcc_l4_32_x_shader[m](x, a, b, splitk_tmp).dispatch(x / (tile_k / 32))
            << splitk_reduce_shader(x, 32, splitk_tmp, d).dispatch(32);
    }

    void mm_rcc_32_l4_x(int n, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d, BufferView<half4> splitk_tmp) {
        if (!mm_rcc_32_l4_x_shader[n]) {
            Kernel1D mm_rcc_32_l4_x = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> splitk_tmp) {
                mm_32_l4_x_impl(n, x, a, b, splitk_tmp);
            };
            mm_rcc_32_l4_x_shader[n] = global::device().compile(mm_rcc_32_l4_x);
        }
        if (!splitk_reduce_shader) {
            Kernel1D splitk_reduce = []($uint x, $uint stride, $buffer<half4> splitk_tmp, $buffer<half4> store) {
                splitk_reduce_impl(x, stride, splitk_tmp, store);
            };
            splitk_reduce_shader = global::device().compile(splitk_reduce);
        }
        global::cmd_list()
            << mm_rcc_32_l4_x_shader[n](x, a, b, splitk_tmp).dispatch(x / (tile_k / 32))
            << splitk_reduce_shader(x, n*8, splitk_tmp, d).dispatch(n*8);
    }
}