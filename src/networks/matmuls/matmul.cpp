#include <luisa/dsl/sugar.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>
#include <global.h>
#include <matmul.h>
#include <activation_func.h>

using namespace luisa;
using namespace luisa::compute;

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

// void add_tile($buffer<half4> &buf, $shared<half4> &smem, $half4 *acc, $uint stride) {
void add_tile($buffer<half4> &buf, $shared<half4> &smem, $array<half4, 16> &acc, $uint stride, $uint tid, $uint bid) {
    // $uint tid = $dispatch_x % 256;
    // $uint bid = $dispatch_x / 256;
    $uint x_ofs = tid%32/8;
    $uint y_ofs = tid%8 + tid/32*16;
    // for (int tx = 0; tx < 2; tx++) {
    //     for (int i = 0; i < 4; i++) {
    //         for (int ty = 0; ty < 2; ty++) {
    //             acc[i + (ty + tx*2)*4] += buf.read(y_ofs + ty*8 + bid*128 + (i + (x_ofs + tx*4)*4)*stride/4);
    //             // acc[i + (ty + tx*2)*4] += $half4(tx,ty,i,1);
    //         }
    //     }
    // }
    // for (int t = 0; t < 2; t++) {
    //     for (int i = 0; i < 4; i++) {
    $for (t, 2) {
        $for (i, 4) {
            sync_block();
            smem[tid] = buf.read(tid%128 + bid*128 + (i + t*16 + tid/128*4)*stride/4);
            smem[tid + 256] = buf.read(tid%128 + bid*128 + (i + t*16 + (tid/128 + 2)*4)*stride/4);
            sync_block();
            acc[i + t*2*4] += smem[y_ofs + x_ofs*128];
            acc[i + (t*2 + 1)*4] += smem[y_ofs + 8 + x_ofs*128];
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
    // $uint *act
    activation::Activation *act
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
        add_tile(*c, b_tile, acc, x, tid, bid);
    }
    if (act != nullptr) {
        // activation::apply_forward(*act, acc);
        $for (i, 16) {
            $for (j, 4) {
                activation::forward(*act, acc[i][j]);
            };
        };
    }
    store_res_tile(d, b_tile, acc, x, tid, bid);
}

namespace matmuls {
    using mm_shader_t = Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>>;
    using mma_shader_t = Shader1D<uint, Buffer<half4>, Buffer<half4>, Buffer<half4>, Buffer<half4>>;

    mm_shader_t act_mm_crr_32_x_32_shader[activation::NUM];
    mm_shader_t act_mm_rrr_32_x_32_shader[activation::NUM];
    mma_shader_t act_mma_crr_32_x_32_shader[activation::NUM];
    mma_shader_t act_mma_rrr_32_x_32_shader[activation::NUM];

    void mm_crr_32_x_32(uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d) {
        act_mm_crr_32_x_32(activation::None, x, a, b, d);
    }

    void mm_rrr_32_x_32(uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d) {
        act_mm_rrr_32_x_32(activation::None, x, a, b, d);
    }

    void mma_crr_32_x_32(uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> c, BufferView<half4> d) {
        act_mma_crr_32_x_32(activation::None, x, a, b, c, d);
    }

    void mma_rrr_32_x_32(uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> c, BufferView<half4> d) {
        act_mma_rrr_32_x_32(activation::None, x, a, b, c, d);
    }

    // matmul and activation result
    void act_mm_crr_32_x_32(activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d) {
        if (!act_mm_crr_32_x_32_shader[act]) {
            Kernel1D act_mm_crr_32_x_32 = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> d) {
                mm_32_x_32_impl(x, a, false, b, nullptr, d, &act);
            };
            act_mm_crr_32_x_32_shader[act] = global::device().compile(act_mm_crr_32_x_32);
        }
        // global::stream() << act_mm_crr_32_x_32_shader((uint)act, x, a, b, d).dispatch(x / 2);
        global::stream() << act_mm_crr_32_x_32_shader[act](x, a, b, d).dispatch(x / 2);
    }

    void act_mm_rrr_32_x_32(activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> d) {
        if (!act_mm_rrr_32_x_32_shader[act]) {
            Kernel1D act_mm_rrr_32_x_32 = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> d) {
                mm_32_x_32_impl(x, a, true, b, nullptr, d, &act);
            };
            act_mm_rrr_32_x_32_shader[act] = global::device().compile(act_mm_rrr_32_x_32);
        }
        // global::stream() << act_mm_rrr_32_x_32_shader((uint)act, x, a, b, d).dispatch(x / 2);
        global::stream() << act_mm_rrr_32_x_32_shader[act](x, a, b, d).dispatch(x / 2);
    }

    void act_mma_crr_32_x_32(activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> c, BufferView<half4> d) {
        if (!act_mma_crr_32_x_32_shader[act]) {
            Kernel1D act_mma_crr_32_x_32 = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> c, $buffer<half4> d) {
                mm_32_x_32_impl(x, a, false, b, &c, d, &act);
            };
            act_mma_crr_32_x_32_shader[act] = global::device().compile(act_mma_crr_32_x_32);
        }
        // global::stream() << act_mma_crr_32_x_32_shader((uint)act, x, a, b, c, d).dispatch(x / 2);
        global::stream() << act_mma_crr_32_x_32_shader[act](x, a, b, c, d).dispatch(x / 2);
    }

    void act_mma_rrr_32_x_32(activation::Activation act, uint x, BufferView<half4> a, BufferView<half4> b, BufferView<half4> c, BufferView<half4> d) {
        if (!act_mma_rrr_32_x_32_shader[act]) {
            Kernel1D act_mma_rrr_32_x_32 = [&]($uint x, $buffer<half4> a, $buffer<half4> b, $buffer<half4> c, $buffer<half4> d) {
                mm_32_x_32_impl(x, a, true, b, &c, d, &act);
            };
            act_mma_rrr_32_x_32_shader[act] = global::device().compile(act_mma_rrr_32_x_32);
        }
        // global::stream() << act_mma_rrr_32_x_32_shader((uint)act, x, a, b, c, d).dispatch(x / 2);
        global::stream() << act_mma_rrr_32_x_32_shader[act](x, a, b, c, d).dispatch(x / 2);
    }
}