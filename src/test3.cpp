#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace fmt;

int main(int argc, char *argv[]) {
    Context ctx{argv[0]};
    Device device = ctx.create_device("dx");
    Stream stream = device.create_stream();

    const uint wmma_m = 16;
    const uint wmma_n = 16;
    const uint wmma_k = 16;

    vector<uint> a_h(wmma_m * wmma_k);
    vector<uint> b_h(wmma_n * wmma_k);
    vector<uint> c_h(wmma_m * wmma_n);
    vector<uint> c_buffer(wmma_m * wmma_n);

    srand(0);
    for (int i = 0; i < 16*16; i++) {
        a_h[i] = rand() % 16;
        b_h[i] = rand() % 16;
    }

    Clock timer;

    timer.tic();
    for (int k = 0; k < wmma_k; k++) {
        for (int y = 0; y < wmma_n; y++) {
            for (int x = 0; x < wmma_m; x++) {
                c_h[x + y*wmma_k] += a_h[x + k*wmma_m] * b_h[y + k*wmma_n];
            }
        }
    }

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            print("{:10d} ", c_h[i + j * 16]);
        }
        print("\n");
    }
    // print("{}\n", timer.toc());
    // print("[");
    // for (int i = 0; i < 32; i++) {
    //     print("{}", c_h[i]);
    //     if (i < 31) print(", ");
    // }
    // print("]\n");

    auto a = device.create_buffer<uint>(wmma_m * wmma_k);
    auto b = device.create_buffer<uint>(wmma_n * wmma_k);
    auto c = device.create_buffer<uint>(wmma_m * wmma_n);

    stream << a.copy_from(a_h.data()) << b.copy_from(b_h.data()) << synchronize();

    Kernel1D kernel1 = [&]() {
        set_block_size(32);
        $array<uint, 8> a_frag;
        $array<uint, 8> b_frag;
        $array<uint, 8> c_frag;

        $ lid = $dispatch_x & 31;

        for (int i = 0; i < 8; i++)
            a_frag[i] = a->read((lid & 15) + (i + lid / 16 * 8) * 16);
        for (int i = 0; i < 8; i++)
            b_frag[i] = b->read((lid & 15) + (i + lid / 16 * 8) * 16);
        for (int i = 0; i < 8; i++)
            c_frag[i] = 0;

        $uint tmp0;
        $uint tmp1;
        for (int t = 0; t < 2; t++) {
            for (int ofs = 0; ofs < 8; ofs++) {
                tmp0 = 0;
                for (int i = 0; i < 8; i++) {
                    tmp1 = warp_read_lane(b_frag[i], lid ^ (ofs ^ (8 * t)));
                    tmp0 += a_frag[i] * tmp1;
                }
                tmp1 = warp_read_lane(tmp0, lid ^ 16);
                $uint op = ((lid >> 3) ^ (lid >> 4)) & 1;
                c_frag[(lid ^ ofs) & 7] += (op ^ (1 - t)) * (tmp0 + tmp1);
            }
        }

        // $uint tmp0;
        // $uint tmp1;
        // for (int ofs = 0; ofs < 8; ofs++) {
        //     tmp0 = 0;
        //     for (int i = 0; i < 8; i++) {
        //         tmp1 = warp_read_lane_at(b_frag[i], lid ^ ofs);
        //         tmp0 += a_frag[i] * tmp1;
        //     }
        //     tmp1 = warp_read_lane_at(tmp0, lid ^ 16);
        //     c_frag[(lid ^ ofs) & 7] += ((~(lid >> 3) ^ (lid >> 4)) & 1) * (tmp0 + tmp1);
        // }
        // for (int ofs = 0; ofs < 8; ofs++) {
        //     tmp0 = 0;
        //     for (int i = 0; i < 8; i++) {
        //         tmp1 = warp_read_lane_at(b_frag[i], lid ^ ofs ^ 8);
        //         tmp0 += a_frag[i] * tmp1;
        //     }
        //     tmp1 = warp_read_lane_at(tmp0, lid ^ 16);
        //     c_frag[(lid ^ ofs) & 7] += (((lid >> 3) ^ (lid >> 4)) & 1) * (tmp0 + tmp1);
        // }
        
        for (int i = 0; i < 8; i++)
            c->write((lid & 15) + (i + lid / 16 * 8) * 16, c_frag[i]);
    };

    auto shader1 = device.compile(kernel1);
    stream << shader1().dispatch(32) << synchronize();

    stream << c.copy_to(c_buffer.data()) << synchronize();

    // uint a_frag[32][8];
    // uint b_frag[32][8];
    // uint c_frag[32][8];
    // uint shlf_reg[32];
    // uint tmp0[32], tmp1[32];

    // {
    //     for (int lid = 0; lid < 32; lid++) {
    //         for (int i = 0; i < 8; i++) {
    //             a_frag[lid][i] = a_h[(lid & 15) + (i + lid / 16 * 8) * 16];
    //             b_frag[lid][i] = b_h[(lid & 15) + (i + lid / 16 * 8) * 16];
    //             c_frag[lid][i] = 0;
    //         }
    //     }

    //     print("a_frag:\n");
    //     for (int i = 0; i < 8; i++) {
    //         for (int j = 0; j < 32; j++) {
    //             print("{:5d} ", a_frag[j][i]);
    //         }
    //         print("\n");
    //     }
    //     print("b_frag:\n");
    //     for (int i = 0; i < 8; i++) {
    //         for (int j = 0; j < 32; j++) {
    //             print("{:5d} ", b_frag[j][i]);
    //         }
    //         print("\n");
    //     }

    //     for (int ofs = 0; ofs < 8; ofs++) {
    //         for (int lid = 0; lid < 32; lid++) tmp0[lid] = 0; 
    //         for (int i = 0; i < 8; i++) {
    //             for (int lid = 0; lid < 32; lid++) shlf_reg[lid] = b_frag[lid][i];
    //             for (int lid = 0; lid < 32; lid++) tmp1[lid] = shlf_reg[lid ^ ofs];
    //             for (int lid = 0; lid < 32; lid++) tmp0[lid] += a_frag[lid][i] * tmp1[lid];
    //         }
    //         for (int lid = 0; lid < 32; lid++) shlf_reg[lid] = tmp0[lid];
    //         for (int lid = 0; lid < 32; lid++) tmp1[lid] = shlf_reg[lid ^ 16];
    //         for (int lid = 0; lid < 32; lid++) c_frag[lid][(lid ^ ofs) & 7] += ((~(lid >> 3) ^ (lid >> 4)) & 1) * (tmp0[lid] + tmp1[lid]);

    //         // for (int lid = 0; lid < 32; lid++) {
    //         //         print("{:5d} ", ((lid ^ ofs) & 7));
    //         //     }
    //         //     print("\n");
    //     }

    //     for (int i = 0; i < 8; i++) {
    //         for (int lid = 0; lid < 32; lid++) c_buffer[(lid & 15) + (i + lid / 16 * 8) * 16] = c_frag[lid][i];
    //     }

    //     // for (int i = 0; i < 8; i++) {
    //     //     for (int j = 0; j < 32; j++) {
    //     //         print("{:5d} ", c_frag[j][i]);
    //     //     }
    //     //     print("\n");
    //     // }
        
    // }

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            print("{:10d} ", c_buffer[i + j * 16]);
        }
        print("\n");
    }
    return 0;
}