#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

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

template <typename T>
$uint size_byte($uint num_element) {
    return (uint)sizeof(T) * num_element;
}

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

void load_32x2($bytebuffer &buf, $shared<half4> &smem) {
    $uint tid = $dispatch_x % 256;
    $if (tid < 16) {
        smem[tid] = buf.read<half4>(size_byte<half4>(tid));
    };
}
void load_transpose_2x32($bytebuffer &buf, $shared<half4> &smem) {
    $uint tid = $dispatch_x % 256;
    $if (tid < 16) {
        $half4 tmp = buf.read<half4>(size_byte<half4>(tid));
        for (int i = 0; i < 4; i++) {
            smem[tid%8*4 + i][tid/8] = tmp[i];
        }
    };
}
void load_32x3($bytebuffer &buf, $shared<half4> &smem) {
    $uint tid = $dispatch_x % 256;
    $if (tid < 24) {
        smem[tid] = buf.read<half4>(size_byte<half4>(tid));
    };
}
void load_transpose_3x32($bytebuffer &buf, $shared<half4> &smem) {
    $uint tid = $dispatch_x % 256;
    $if (tid < 24) {
        $half4 tmp = buf.read<half4>(size_byte<half4>(tid));
        for (int i = 0; i < 4; i++) {
            smem[tid%8*4 + i][tid/8] = tmp[i];
        }
    };
}
void load_32x32($bytebuffer &buf, $shared<half4> &smem) {
    $uint tid = $dispatch_x % 256;
    smem[tid] = buf.read<half4>(size_byte<half4>(tid));
}
void load_transpose_32x32($bytebuffer &buf, $shared<half4> &smem) {
    $uint tid = $dispatch_x % 256;
    $half4 tmp = buf.read<half4>(size_byte<half4>(tid));
    for (int i = 0; i < 4; i++) {
        smem[tid/32 + (tid%8*4 + i)*8][tid/8%4] = tmp[i];
    }
}
void load_transpose_512x2($bytebuffer &buf, $shared<half4> &smem, $uint offset) {
    $uint tid = $dispatch_x % 256;
    $half4 tmp = buf.read<half4>(size_byte<half4>(tid + offset/4));
    for (int i = 0; i < 4; i++) {
        smem[tid/2 + i%2*128][i/2 + tid%2*2] = tmp[i];
    }
}
void load_512x3($bytebuffer &buf, $shared<half4> &smem, $uint offset, $uint stride) {
    $uint tid = $dispatch_x % 256;
    smem[tid] = buf.read<half4>(size_byte<half>(offset + tid/128*stride));
    $if (tid < 128) {
        smem[tid + 256] = buf.read<half4>(size_byte<half>(offset + (2 + tid/128)*stride));
    };
}
void load_512x4($bytebuffer &buf, $shared<half4> &smem, $uint offset, $uint stride) {
    $uint tid = $dispatch_x % 256;
    smem[tid] = buf.read<half4>(size_byte<half>(offset + tid/128*stride));
    smem[tid + 256] = buf.read<half4>(size_byte<half>(offset + (2 + tid/128)*stride));
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

const uint input_dim = 2;
const uint output_dim = 3;
const uint output_dim_pad = 4;
const uint layer_width = 32;
const uint batch_size = 1920 * 1080;
// const uint batch_size = 16384;

const uint block_size = 256;

enum class Activation {
	ReLU,
	LeakyReLU,
	Sine,
	None,
};
const Activation activation = Activation::Sine;

// bool test_activation = false;

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
        case Activation::Sine: x *= sin(asin(fwd_act)).cast<half>();break;
        case Activation::None:break;
    }
}
void apply_activation_backward($half4 &acc, $half4 &fwd_act) {
    for (int i = 0; i < 4; i++) {
        activation_backward(acc[i], fwd_act[i]);
    }
}

Kernel1D output_forward_kernel = []($bytebuffer weight, $bytebuffer act, $bytebuffer store) {
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
        b_frag = act.read<half4>(size_byte<half4>(tid + bid*256 + k*batch_size/4));
        out_product_3x4_c(a_frag, b_frag, acc);
    };
    for (int i = 0; i < 4; i++) {
        store.write(size_byte<half4>(i + tid*4 + bid*256*4), acc[i]);
    }
};

Kernel1D input_backward_kernel = []($bytebuffer weight, $bytebuffer grad, $bytebuffer store) {
    set_block_size(256);
    $shared<half4> a_tile{32};
    $half4 acc[2];
    $half4 a_frag;
    $half4 b_frag;

    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;

    load_transpose_2x32(weight, a_tile);
    sync_block();
    $for (k, layer_width) {
        a_frag = a_tile[k];
        b_frag = grad.read<half4>(size_byte<half4>(tid + bid*256 + k*batch_size/4));
        out_product_2x4_r(a_frag, b_frag, acc);
    };
    for (int i = 0; i < 2; i++) {
        store.write(size_byte<half4>(tid + bid*256 + i*batch_size/4), acc[i]);
    }
};

// 放弃挣扎
void store_res_tile($bytebuffer &buf, $shared<half4> &smem, $half4 *acc, $uint stride, $uint tid, $uint bid) {
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

            buf.write(size_byte<half4>(tid%128 + bid*128 + (i + t*16 + tid/128*4)*stride/4), smem[tid]);
            buf.write(size_byte<half4>(tid%128 + bid*128 + (i + t*16 + (tid/128 + 2)*4)*stride/4), smem[tid + 256]);
        }
    }
}

// 放弃挣扎
void apply_activation_backward_tile($bytebuffer &buf, $shared<half4> &smem, $bytebuffer &fwd_act, $half4 *acc, $uint stride, $uint tid, $uint bid) {
    // $uint tid = $dispatch_x % 256;
    // $uint bid = $dispatch_x / 256;
    $uint x_ofs = tid%32/8;
    $uint y_ofs = tid%8 + tid/32*16;
    $half4 tmp;
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 4; i++) {
            sync_block();
            smem[tid] = fwd_act.read<half4>(size_byte<half4>(tid%128 + bid*128 + (i + t*16 + tid/128*4)*stride/4));
            smem[tid + 256] = fwd_act.read<half4>(size_byte<half4>(tid%128 + bid*128 + (i + t*16 + (tid/128 + 2)*4)*stride/4));
            sync_block();
            tmp = smem[y_ofs + x_ofs*128];
            apply_activation_backward(acc[i + t*2*4], tmp);
            tmp = smem[y_ofs + 8 + x_ofs*128];
            apply_activation_backward(acc[i + (t*2 + 1)*4], tmp);
        }
    }
}

Kernel1D output_backward_kernel = []($bytebuffer weight, $bytebuffer grad, $bytebuffer fwd_act, $bytebuffer store) {
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
    load_512x3(grad, b_tile, (tid%128 + bid*128)*4, batch_size);
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

void layer_kernel_impl($bytebuffer &weight, $bytebuffer &buf, $bytebuffer &store, bool is_backward, $bytebuffer *fwd_act) {
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
        load_512x4(buf, b_tile, (tid%128 + bid*128)*4 + k*batch_size, batch_size);
        sync_block();
        $for (k1, 4) {
            for (int t = 0; t < 2; t++) {
                a_frag[t] = a_tile[x_ofs + t*4 + (k + k1)*8];
                b_frag[t] = b_tile[y_ofs + t*8 + k1*128];
            }
            out_product_8x8_r(a_frag, b_frag, acc);
        };
    };
    if(!is_backward) {
        for (int i = 0; i < 16; i++) {
            apply_activation_forward(acc[i]);
        }
    }
    else {
        apply_activation_backward_tile(store, b_tile, *fwd_act, acc, batch_size, tid, bid);
    }
    store_res_tile(store, b_tile, acc, batch_size, tid, bid);
}

Kernel1D layer_forward_kernel = []($bytebuffer weight, $bytebuffer act, $bytebuffer store) {
    layer_kernel_impl(weight, act, store, false, nullptr);
};
Kernel1D layer_backward_kernel = []($bytebuffer weight, $bytebuffer grad, $bytebuffer fwd_act, $bytebuffer store) {
    layer_kernel_impl(weight, grad, store, true, &fwd_act);
};

Kernel1D input_forward_kernel = []($bytebuffer weight, $bytebuffer input, $bytebuffer store) {
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
    load_transpose_512x2(input, b_tile, bid*1024);
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


int main(int argc, char *argv[]) {
    lc::init(argv[0]);
    Clock timer;

    enum {
        output_forward,
        output_backward,
        layer_forward,
        layer_backward,
        input_forward,
        input_backward
    } test_case = input_backward;

    uint a_size = 0;
    uint b_size = 0;
    uint c_size = 0;
    uint fwd_act_size = 0;

    switch (test_case) {
        case output_forward: {
            a_size = output_dim * layer_width;
            b_size = layer_width * batch_size;
            c_size = output_dim_pad * batch_size;
        } break;
        case output_backward: {
            a_size = output_dim * layer_width;
            b_size = output_dim * batch_size;
            c_size = layer_width * batch_size;
            fwd_act_size = c_size;
        } break;
        case layer_forward: {
            a_size = layer_width * layer_width;
            b_size = layer_width * batch_size;
            c_size = layer_width * batch_size;
        } break;
        case layer_backward: {
            a_size = layer_width * layer_width;
            b_size = layer_width * batch_size;
            c_size = layer_width * batch_size;
            fwd_act_size = c_size;
        } break;
        case input_forward: {
            a_size = input_dim * layer_width;
            b_size = input_dim * batch_size;
            c_size = layer_width * batch_size;
        } break;
        case input_backward: {
            a_size = input_dim * layer_width;
            b_size = layer_width * batch_size;
            c_size = input_dim * batch_size;
        } break;
    }

    auto a = lc::device.create_byte_buffer(a_size * sizeof(half));
    auto b = lc::device.create_byte_buffer(b_size * sizeof(half));
    auto c = lc::device.create_byte_buffer(c_size * sizeof(half));
    auto fwd_act = lc::device.create_byte_buffer(c_size * sizeof(half));

    vector<half> a_h(a_size);
    vector<half> b_h(b_size);
    vector<float> c_h(c_size);
    vector<half> c_buffer(c_size);
    vector<half> fwd_act_h(fwd_act_size);

    for (auto &x : a_h) x = (pcg32::next_float() * 2 - 1) * 0.3;
    for (auto &x : b_h) x = pcg32::next_float();
    for (auto &x : fwd_act_h) x = pcg32::next_float() - 0.5;

    timer.tic();

    switch (test_case) {
        case output_forward: {
            for (int x = 0; x < output_dim; x++) {
                for (int k = 0; k < layer_width; k++) {
                    float tmp = a_h[k + x*layer_width];
                    for (int y = 0; y < batch_size; y++) {
                        c_h[x + y*output_dim_pad] += tmp * b_h[y + k*batch_size];
                    }
                }
            }
        } break;
        case output_backward: {
            for (int x = 0; x < layer_width; x++) {
                for (int k = 0; k < output_dim; k++) {
                    float tmp = a_h[x + k*layer_width];
                    for (int y = 0; y < batch_size; y++) {
                        c_h[y + x*batch_size] += tmp * b_h[y + k*batch_size];
                    }
                }
            }
        } break;
        case layer_forward: {
            for (int k = 0; k < layer_width; k++) {
                for (int x = 0; x < layer_width; x++) {
                    float tmp = a_h[x + k*layer_width];
                    for (int y = 0; y < batch_size; y++) {
                        c_h[y + x*batch_size] += tmp * b_h[y + k*batch_size];
                    }
                }
            }
        } break;
        case layer_backward: {
            for (int x = 0; x < layer_width; x++) {
                for (int k = 0; k < layer_width; k++) {
                    float tmp = a_h[k + x*layer_width];
                    for (int y = 0; y < batch_size; y++) {
                        c_h[y + x*batch_size] += tmp * b_h[y + k*batch_size];
                    }
                }
            }
        } break;
        case input_forward: {
            for (int k = 0; k < input_dim; k++) {
                for (int x = 0; x < layer_width; x++) {
                    float tmp = a_h[x + k*layer_width];
                    for (int y = 0; y < batch_size; y++) {
                        c_h[y + x*batch_size] += tmp * b_h[k + y*input_dim];
                    }
                }
            }
        } break;
        case input_backward: {
            for (int x = 0; x < input_dim; x++) {
                for (int k = 0; k < layer_width; k++) {
                    float tmp = a_h[k + x*layer_width];
                    for (int y = 0; y < batch_size; y++) {
                        c_h[y + x*batch_size] += tmp * b_h[y + k*batch_size];
                    }
                }
            }
        } break;
    }
    
    switch (test_case) {
        case input_forward:
        case layer_forward: {
            for (float &x : c_h) {
                activation_forward_host(x);
            }
        } break;
        case output_backward:
        case layer_backward: {
            int i = 0;
            for (float &x : c_h) {
                activation_backward_host(x, fwd_act_h[i]);
                i++;
            }
        } break;
        case output_forward:
        case input_backward:
            break;
    }

    print("calc ref res in: {}\n", timer.toc());

    print("c_h: [");
    for (int i = 0; i < 32; i++) {
        print("{}", c_h[i]);
        if (i < 31) print(", ");
    }
    print("]\n");

    timer.tic();
    Shader1D<ByteBuffer, ByteBuffer, ByteBuffer> shader1;
    Shader1D<ByteBuffer, ByteBuffer, ByteBuffer, ByteBuffer> shader2;
    switch (test_case) {
        case output_forward: {
            shader1 = lc::device.compile(output_forward_kernel);
        } break;
        case output_backward: {
            shader2 = lc::device.compile(output_backward_kernel);
        } break;
        case layer_forward: {
            shader1 = lc::device.compile(layer_forward_kernel);
        } break;
        case layer_backward: {
            shader2 = lc::device.compile(layer_backward_kernel);
        } break;
        case input_forward: {
            shader1 = lc::device.compile(input_forward_kernel);
        } break;
        case input_backward: {
            shader1 = lc::device.compile(input_backward_kernel);
        } break;
    }
    print("compiled shader: {}\n", timer.toc());

    lc::stream << a.copy_from(a_h.data()) << b.copy_from(b_h.data());
    if (test_case == output_backward || test_case == layer_backward) {
        lc::stream << fwd_act.copy_from(fwd_act_h.data());
    }
    lc::stream.synchronize();

    for (int i = 0; i < 10; i++) {
        timer.tic();
        for (int i = 0; i < 100; i++) {
            switch (test_case) {
                case output_forward: {
                    lc::stream << shader1(a, b, c).dispatch(batch_size / 4);
                } break;
                case input_backward: {
                    lc::stream << shader1(a, b, c).dispatch(batch_size / 4);
                } break;
                case output_backward:
                case layer_backward: {
                    lc::stream << shader2(a, b, fwd_act, c).dispatch(batch_size / 2);
                } break;
                case layer_forward:
                case input_forward: {
                    lc::stream << shader1(a, b, c).dispatch(batch_size / 2);
                } break;
            }
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
    print("]\n");

    float f_err = 0;
    float f_err2 = 0;
    int err_c = 0;
    int i = 0;
    for (float x: c_buffer) {
        float y = c_h[i];
        float err = abs(y - x) / max(abs(x), abs(y));
        float err2 = abs(y - x);

        if (err > f_err) {
            print("!inc error {}: {}, {}; f_err: {}\n", i, y, x, err);
        }

        f_err = max(f_err, err);
        f_err2 = max(f_err2, err2);
        if (err > 0.005 || err2 > 0.05) {
            if (err_c < 32) {
                print("error {}: {}, {}\n", i, y, x);
            }
            err_c++;
        }
        i++;
    }
    print("f_err: {}\n", f_err);
    print("f_err2: {}\n", f_err2);
    print("err_c: {}\n", err_c);
    print("ok\n");
    return 0;
}