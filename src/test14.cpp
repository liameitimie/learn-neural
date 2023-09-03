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

const uint input_dim = 2;
const uint output_dim = 3;
const uint output_dim_pad = 4;

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

const uint layer_width = 32;
const uint hidden_layers = 5;

ByteBuffer layer_weight[hidden_layers - 1];
ByteBuffer input_weight;
ByteBuffer output_weight;

void init_buffer() {
    for (int i = 0; i < hidden_layers - 1; i++) {
        layer_weight[i] = lc::device.create_byte_buffer(layer_width * layer_width * sizeof(float));
    }
    input_weight = lc::device.create_byte_buffer(input_dim * layer_width * sizeof(float));
    output_weight = lc::device.create_byte_buffer(output_dim_pad * layer_width * sizeof(float));
}

template <typename T>
$uint size_byte($uint num_element) {
    return (uint)sizeof(T) * num_element;
}

const uint batch_size = 1920 * 1080;

Kernel1D output_forward_kernel = []($bytebuffer weight, $bytebuffer act, $bytebuffer store) {
    set_block_size(256);
    $shared<float4> a_tile{32};
    $float4 acc[4];
    $float4 a_frag;
    $float4 b_frag;

    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;

    // $if (tid < 32) {
    //     a_tile[tid] = weight.read<float4>(size_byte<float4>(tid));
    // };
    $if (tid < 96) {
        a_tile[tid%32][tid/32] = weight.read<float>(size_byte<float>(tid));
    };
    sync_block();
    $for (k, layer_width) {
        a_frag = a_tile[k];
        b_frag = act.read<float4>(size_byte<float4>(tid + bid*256 + k*batch_size/4));
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 3; i++) {
                acc[j][i] += a_frag[i] * b_frag[j];
            }
        }
    };
    for (int i = 0; i < 4; i++) {
        store.write(size_byte<float4>(i + tid*4 + bid*256*4), acc[i]);
    }
};
Kernel1D output_backward_kernel = []($bytebuffer weight, $bytebuffer grad, $bytebuffer store) {
    set_block_size(256);
    $shared<float4> a_tile{24};
    $shared<float4> b_tile{64};
    $shared<float4> c_tile{256};
    $float4 acc[8];
    $float4 a_frag;
    $float4 b_frag[2];

    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;

    $uint x_ofs = tid % 8;
    $uint y_ofs = tid / 8;
    // $uint x_ofs = tid % 32 / 4;
    // $uint y_ofs = tid % 4 + tid / 32 * 4;

    $if (tid < 24) {
        a_tile[tid] = weight.read<float4>(size_byte<float4>(tid));
    };
    $for (k, output_dim) {
        sync_block();
        b_tile[tid/4][tid%4] = grad.read<float>(size_byte<float>(tid + bid*256 + k*batch_size));
        sync_block();
        a_frag = a_tile[x_ofs + k*8];
        b_frag[0] = b_tile[y_ofs];
        b_frag[1] = b_tile[y_ofs + 32];
        for (int t = 0; t < 2; t++) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    acc[i + t*4][j] += a_frag[i] * b_frag[t][j];
                }
            }
        }
    };
    for (int i = 0; i < 8; i++) {
        sync_block();
        c_tile[y_ofs + x_ofs*32] = acc[i];
        sync_block();
        store.write(size_byte<float4>(tid%32 + i/4*32 + bid*256/4 + (tid/32*4 + i%4)*batch_size/4), c_tile[tid]);

        // store.write(
        //     size_byte<float4>(y_ofs + i%2*32 + bid*256/4 + (x_ofs*4 + i/2)*batch_size/4), 
        //     acc[i/2 + i%2*4]
        // );
    }
};

Kernel1D output_backward_kernel1 = []($bytebuffer weight, $bytebuffer grad, $bytebuffer store) {
    set_block_size(256);
    $shared<float4> a_tile{24};
    $shared<float4> b_tile{512};
    $float4 acc[2][2][4];
    $float4 a_frag[2];
    $float4 b_frag[2];

    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;

    $uint x_ofs = tid%32/8;
    $uint y_ofs = tid%8 + tid/32*16;

    $if (tid < 24) {
        a_tile[tid] = weight.read<float4>(size_byte<float4>(tid));
    };
    b_tile[tid] = grad.read<float4>(size_byte<float4>(tid%128 + bid*128 + tid/128*batch_size/4));
    $if (tid < 128) {
        b_tile[tid + 256] = grad.read<float4>(size_byte<float4>(tid%128 + bid*128 + (2 + tid/128)*batch_size/4));
    };
    sync_block();

    $for (k, output_dim) {
        for (int t = 0; t < 2; t++) {
            a_frag[t] = a_tile[x_ofs + t*4 + k*8];
            b_frag[t] = b_tile[y_ofs + t*8 + k*128];
        }
        for (int tx = 0; tx < 2; tx++) {
            for (int ty = 0; ty < 2; ty++) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        acc[tx][ty][i][j] += a_frag[tx][i] * b_frag[ty][j];
                    }
                }
            }
        }
    };
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 4; i++) {
            sync_block();
            b_tile[y_ofs + x_ofs*128] = acc[t][0][i];
            b_tile[y_ofs + 8 + x_ofs*128] = acc[t][1][i];
            sync_block();

            store.write(size_byte<float4>(tid%128 + bid*128 + (i + t*16 + tid/128*4)*batch_size/4), b_tile[tid]);
            store.write(size_byte<float4>(tid%128 + bid*128 + (i + t*16 + (tid + 256)/128*4)*batch_size/4), b_tile[tid + 256]);
        }
    }
};

const uint tile_width = 256;
const uint tile_size = layer_width * tile_width;

Kernel1D layer_forward_kernel = []($bytebuffer weight, $bytebuffer act, $bytebuffer store) {
    set_block_size(256);
    $shared<float4> a_tile{256};
    $shared<float4> b_tile{64};
    $shared<float4> c_tile{256};
    $float4 acc[8];
    $float4 a_frag;
    $float4 b_frag[2];

    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;

    $uint x_ofs = tid % 8;
    $uint y_ofs = tid / 8;

    a_tile[tid] = weight.read<float4>(size_byte<float4>(tid));
    $for (k, layer_width) {
        sync_block();
        b_tile[tid/4][tid%4] = act.read<float>(size_byte<float>(tid + k*256 + bid*tile_size));
        sync_block();
        a_frag = a_tile[x_ofs + k*8];
        b_frag[0] = b_tile[y_ofs];
        b_frag[1] = b_tile[y_ofs + 32];
        for (int t = 0; t < 2; t++) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    acc[i + t*4][j] += a_frag[i] * b_frag[t][j];
                }
            }
        }
    };
    for (int i = 0; i < 8; i++) {
        sync_block();
        c_tile[y_ofs + x_ofs*32] = acc[i];
        sync_block();
        store.write(size_byte<float4>(tid%32 + i/4*32 + (tid/32*4 + i%4)*64 + bid*tile_size/4), c_tile[tid]);
        // store.write(
        //     size_byte<float4>(y_ofs + i%2*32 + bid*256/4 + (x_ofs*4 + i/2)*batch_size/4), 
        //     acc[i/2 + i%2*4]
        // );
    }
};


Kernel1D layer_forward_kernel1 = []($bytebuffer weight, $bytebuffer act, $bytebuffer store) {
    const uint tile_k = 4;
    set_block_size(256);
    $shared<float4> a_tile{256};
    $shared<float4> b_tile{128*tile_k};
    $float4 acc[2][2][4];
    $float4 a_frag[2];
    $float4 b_frag[2];

    $uint tid = $dispatch_x % 256;
    $uint bid = $dispatch_x / 256;

    $uint x_ofs = tid%32/8;
    $uint y_ofs = tid%8 + tid/32*16;

    a_tile[tid] = weight.read<float4>(size_byte<float4>(tid));
    $for (k, 0u, layer_width, tile_k) {
        sync_block();
        // $if (tid < 128) {
        //     b_tile[tid] = act.read<float4>(size_byte<float4>(tid + bid*128 + k*batch_size/4));
        //     b_tile[tid + 128] = act.read<float4>(size_byte<float4>(tid + bid*128 + (k + 1)*batch_size/4));
        // };
        for (int i = 0; i < tile_k/2; i++) {
            b_tile[tid + i*256] = act.read<float4>(size_byte<float4>(tid%128 + bid*128 + (k + (tid + i*256)/128)*batch_size/4));
        }
        // b_tile[tid] = act.read<float4>(size_byte<float4>(tid%128 + bid*128 + (k + tid/128)*batch_size/4));
        // b_tile[tid + 256] = act.read<float4>(size_byte<float4>(tid%128 + bid*128 + (k + (tid+256)/128)*batch_size/4));
        sync_block();
        $for (k1, tile_k) {
        // for (int k1 = 0; k1 < 2; k1++) {
            for (int t = 0; t < 2; t++) {
                a_frag[t] = a_tile[x_ofs + t*4 + (k + k1)*8];
                b_frag[t] = b_tile[y_ofs + t*8 + k1*128];
            }
            for (int tx = 0; tx < 2; tx++) {
                for (int ty = 0; ty < 2; ty++) {
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            acc[tx][ty][i][j] += a_frag[tx][i] * b_frag[ty][j];
                        }
                    }
                }
            }
        };
    };
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 4; i++) {
            if (tile_k < 4) {
                sync_block();
                $if (tid < 128) {
                    a_tile[y_ofs + x_ofs*64] = acc[t][0][i];
                    a_tile[y_ofs + 8 + x_ofs*64] = acc[t][1][i];
                } $else {
                    b_tile[y_ofs - 64 + x_ofs*64] = acc[t][0][i];
                    b_tile[y_ofs - 64 + 8 + x_ofs*64] = acc[t][1][i];
                };
                sync_block();

                store.write(size_byte<float4>(tid%64 + bid*128 + (i + t*16 + tid/64*4)*batch_size/4), a_tile[tid]);
                store.write(size_byte<float4>(tid%64 + 64 + bid*128 + (i + t*16 + tid/64*4)*batch_size/4), b_tile[tid]);
            }
            else {
                sync_block();
                b_tile[y_ofs + x_ofs*128] = acc[t][0][i];
                b_tile[y_ofs + 8 + x_ofs*128] = acc[t][1][i];
                sync_block();

                store.write(size_byte<float4>(tid%128 + bid*128 + (i + t*16 + tid/128*4)*batch_size/4), b_tile[tid]);
                store.write(size_byte<float4>(tid%128 + bid*128 + (i + t*16 + (tid + 256)/128*4)*batch_size/4), b_tile[tid + 256]);
            }
            // store.write(
            //     size_byte<float4>(y_ofs + i%2*32 + bid*256/4 + (x_ofs*4 + i/2)*batch_size/4), 
            //     acc[i/2 + i%2*4]
            // );
        }
    }
};

int main(int argc, char *argv[]) {
    lc::init(argv[0]);
    // init_buffer();

    Clock timer;

    // 1
    auto a = lc::device.create_byte_buffer(output_dim * layer_width * sizeof(float));
    auto b = lc::device.create_byte_buffer(layer_width * batch_size * sizeof(float));
    auto c = lc::device.create_byte_buffer(output_dim_pad * batch_size * sizeof(float));
    // 2
    // auto a = lc::device.create_byte_buffer(output_dim * layer_width * sizeof(float));
    // auto b = lc::device.create_byte_buffer(output_dim * batch_size * sizeof(float));
    // auto c = lc::device.create_byte_buffer(layer_width * batch_size * sizeof(float));
    // 3
    // auto a = lc::device.create_byte_buffer(layer_width * layer_width * sizeof(float));
    // auto b = lc::device.create_byte_buffer(layer_width * batch_size * sizeof(float));
    // auto c = lc::device.create_byte_buffer(layer_width * batch_size * sizeof(float));

    // 1
    vector<float> a_h(output_dim * layer_width);
    vector<float> b_h(layer_width * batch_size);
    vector<float> c_h(output_dim_pad * batch_size);
    vector<float> c_buffer(output_dim_pad * batch_size);
    // 2
    // vector<float> a_h(output_dim * layer_width);
    // vector<float> b_h(output_dim * batch_size);
    // vector<float> c_h(layer_width * batch_size);
    // vector<float> c_buffer(layer_width * batch_size);
    // 3
    // vector<float> a_h(layer_width * layer_width);
    // vector<float> b_h(layer_width * batch_size);
    // vector<float> c_h(layer_width * batch_size);
    // vector<float> c_buffer(layer_width * batch_size);

    for (float &x : a_h) x = pcg32::next_float();
    for (float &x : b_h) x = pcg32::next_float();
    
    timer.tic();

    // 1
    for (int x = 0; x < output_dim; x++) {
        for (int k = 0; k < layer_width; k++) {
            float tmp = a_h[k + x*layer_width];
            for (int y = 0; y < batch_size; y++) {
                c_h[x + y*output_dim_pad] += tmp * b_h[y + k*batch_size];
            }
        }
    }
    // 2
    // for (int x = 0; x < layer_width; x++) {
    //     for (int k = 0; k < output_dim; k++) {
    //         float tmp = a_h[x + k*layer_width];
    //         for (int y = 0; y < batch_size; y++) {
    //             c_h[y + x*batch_size] += tmp * b_h[y + k*batch_size];
    //         }
    //     }
    // }
    // 3
    // for (int t = 0; t < batch_size/tile_width; t++) {
    //     for (int x = 0; x < layer_width; x++) {
    //         for (int k = 0; k < layer_width; k++) {
    //             float tmp = a_h[x + k*layer_width];
    //             for (int y = 0; y < tile_width; y++) {
    //                 c_h[y + x*tile_width + t*tile_size] += tmp * b_h[y + k*tile_width + t*tile_size];
    //             }
    //         }
    //     }
    // }
    // 3.1
    // for (int k = 0; k < layer_width; k++) {
    //     for (int x = 0; x < layer_width; x++) {
    //         float tmp = a_h[x + k*layer_width];
    //         for (int y = 0; y < batch_size; y++) {
    //             c_h[y + x*batch_size] += tmp * b_h[y + k*batch_size];
    //         }
    //     }
    // }

    print("{}\n", timer.toc());
    print("c_h: [");
    for (int i = 0; i < 32; i++) {
        print("{}", c_h[i]);
        if (i < 31) print(", ");
    }
    print("]\n");

    timer.tic();
    auto shader = lc::device.compile(output_forward_kernel);
    // auto shader = lc::device.compile(output_backward_kernel);
    // auto shader = lc::device.compile(output_backward_kernel1);
    // auto shader = lc::device.compile(layer_forward_kernel);
    // auto shader = lc::device.compile(layer_forward_kernel1);
    print("compiled shader: {}\n", timer.toc());

    lc::stream << a.copy_from(a_h.data()) << b.copy_from(b_h.data()) << synchronize();

    for (int i = 0; i < 10; i++) {
        timer.tic();
        for (int i = 0; i < 100; i++) {
            // 1
            lc::stream << shader(a, b, c).dispatch(batch_size / 4);
            // 2
            // lc::stream << shader(a, b, c).dispatch(batch_size);
            // lc::stream << shader(a, b, c).dispatch(batch_size / 2);
            // 3
            // lc::stream << shader(a, b, c).dispatch(batch_size / 2);
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
    int err_c = 0;
    int i = 0;
    // for (int i = 0; i < output_dim_pad * batch_size; i++) {
    for (float x: c_buffer) {
        float err = abs(c_h[i] - x);
        f_err = max(f_err, err);
        if (err > 0.01) {
            if (err_c < 32) {
                print("error {}: {}, {}\n", i, c_h[i], x);
            }
            err_c++;
        }
        i++;
    }
    print("f_err: {}\n", f_err);
    print("err_c: {}\n", err_c);
    print("ok\n");
    return 0;
}