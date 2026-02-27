
#include "globals.hpp"

void average_fixed(float xx[MAX_T*dim_u], float xx_avg[MAX_T*dim_u], int window_size) {

    #pragma HLS inline
    #pragma HLS ARRAY_PARTITION variable=xx complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xx_avg complete dim=1

    const int n = MAX_T;
    const int dim = dim_u;
    const int half = window_size / 2;
    float invW_hp = float(1.0) / float(window_size);

    MAIN_CONV_LOOP: for (int i = 0; i < n; ++i) {
        #pragma HLS PIPELINE II=1  // Reduced from II=2
        
        DIM_LOOP: for (int d = 0; d < dim; ++d) {
           #pragma HLS UNROLL
            
            float sum_hp = 0;
            
            KERNEL_LOOP: for (int k = 0; k < window_size; ++k) {
                // #pragma HLS UNROLL factor=1  // No unroll (sequential)
                
                int j = i + k - half;
                if (j >= 0 && j < n) {
                    sum_hp += (float)xx[j * dim + d] * invW_hp;
                }
            }
            
            xx_avg[i * dim + d] = (float)sum_hp;
        }
    }

    int n_conv = (window_size + 1) / 2;
    BOUNDARY_REGION:{
    #pragma HLS LOOP_MERGE force
    LEFT_BOUNDARY: for (int i = 0; i < n_conv; ++i) {
        #pragma HLS PIPELINE II=1
        
        for (int d = 0; d < dim; ++d) {
            #pragma HLS UNROLL
            int denom = i + n_conv - (window_size % 2);
            if (denom > 0) {
                float factor_hp = float(window_size) / float(denom);
                xx_avg[i * dim + d] = (float)(
                    (float)xx_avg[i * dim + d] * factor_hp
                );
            }
        }
    }

    RIGHT_BOUNDARY: for (int i = 0; i < n_conv; ++i) {
        #pragma HLS PIPELINE II=1
        
        for (int d = 0; d < dim; ++d) {
            #pragma HLS UNROLL
            int right_i = n - 1 - i;
            if (right_i >= n_conv) {
                int denom = i + n_conv - (window_size % 2);
                if (denom > 0) {
                    float factor_hp = float(window_size) / float(denom);
                    xx_avg[right_i * dim + d] = (float)(
                        (float)xx_avg[right_i * dim + d] * factor_hp
                    );
                }
            }
        }
    }
    }

}