#include "globals.hpp"

// void compute_weight_fixed(cost_w S[MAX_K], sample_w W[MAX_K]) {
// #pragma HLS inline

//     #pragma HLS ARRAY_PARTITION variable=S cyclic factor=16 dim=1
//     #pragma HLS ARRAY_PARTITION variable=W cyclic factor=16 dim=1

//     // 1. Find minimum cost S_min via tree reduction
//     cost_w S_min = S[0];
//     FIND_MIN_UNROLLED: for (int k = 0; k < MAX_K; k += 16) {
//         #pragma HLS PIPELINE II=1
//         for (int j = 0; j < 16 && (k + j) < MAX_K; ++j) {
//             // #pragma HLS UNROLL factor=16
//             cost_w v = S[k + j];
//             if (v < S_min) S_min = v;
//         }
//     }

//     // 2. Precompute inverse lambda (fixed-point)
//     const float inv_lambda = (1.0f) / param_lambda;

//     // 3. Compute exponentials and partial sums in fixed-point
//     static weight_t exp_values[MAX_K];
//     #pragma HLS ARRAY_PARTITION variable=exp_values cyclic factor=16 dim=1

//     static weight_t sum_accumulators[16];
//     #pragma HLS ARRAY_PARTITION variable=sum_accumulators complete

//     // Initialize accumulators
//     for (int i = 0; i < 16; ++i) {
//         #pragma HLS UNROLL
//         sum_accumulators[i] = (0);
//     }

//     COMPUTE_EXP_AND_SUM: for (int k = 0; k < MAX_K; ++k) {
//         #pragma HLS PIPELINE 
//         // #pragma HLS UNROLL factor=16

//         // Normalize cost and scale by lambda
//         weight_t diff = S[k] - S_min;
//         weight_t  neg = weight_t(-1) * diff * weight_t(inv_lambda);

//         // Clamp to prevent overflow in exp
//         if (neg > weight_t(20.0f))  neg = weight_t(20.0f);
//         if (neg < weight_t(-20.0f)) neg = weight_t(-20.0f);

//         // Fixed-point exponential (CORDIC or LUT-based)
//         weight_t e = hls::expf(neg);
//         exp_values[k] = e;

//         // 4-way sum
//         sum_accumulators[k & 15] += e;
//     }

//     // 4. Sum reduction
//     // weight_t sum_all = sum_accumulators[0] + sum_accumulators[1]
//                 //   + sum_accumulators[2] + sum_accumulators[3];
//     weight_t sum_all = 0;
//     SUM_ALL_LOOP: for(int i=0; i<16; i++){
//         #pragma HLS PIPELINE
//         sum_all += sum_accumulators[i];
//     }

//     // Avoid divide-by-zero: convert to float for reciprocal then back
//     weight_t sum_f = weight_t(sum_all);
//     weight_t inv_eta_f = (sum_f > weight_t(1e-6f)) ? weight_t(weight_t(1.0f) / sum_f) : weight_t(1e6f);
//     const weight_t inv_eta = weight_t(inv_eta_f);

//     // 5. Normalize weights in fixed-point
//     NORMALIZE: for (int k = 0; k < MAX_K; ++k) {
//         // #pragma HLS PIPELINE II=1
//         #pragma HLS UNROLL factor=16
//         weight_t w_f = exp_values[k] * inv_eta;
//         W[k] = w_f;
//     }
// }

void compute_weight_fixed(cost_w S[MAX_K], sample_w W[MAX_K], float p_lambda) {
    #pragma HLS inline
    
    #pragma HLS ARRAY_PARTITION variable=S cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=W cyclic factor=16 dim=1
    
    // // Find min (sequential is fine, small cost)
    // cost_w S_min = S[0];
    // for (int k = 1; k < MAX_K; ++k) {
    //     #pragma HLS PIPELINE II =2
    //     if (S[k] < S_min) S_min = S[k];
    // }
    // Find min using a parallel reduction tree to shorten the critical path
    cost_w min_vals[MAX_K];
    #pragma HLS ARRAY_PARTITION variable=min_vals complete dim=1
    for(int i=0; i<MAX_K; i++) {
        #pragma HLS UNROLL
        min_vals[i] = S[i];
    }

    // Tree reduction loop
    REDUCE_LOOP: for (int d = MAX_K >> 1; d > 0; d >>= 1) {
        #pragma HLS PIPELINE II=1
        for (int i = 0; i < d; i++) {
            #pragma HLS UNROLL
            cost_w val1 = min_vals[i];
            cost_w val2 = min_vals[i + d];
            if (val2 < val1) {
                min_vals[i] = val2;
            }
        }
    }
    cost_w S_min = min_vals[0];
    
    // Compute exponentials (parallel)
    static weight_t exp_values[MAX_K];
    #pragma HLS ARRAY_PARTITION variable=exp_values cyclic factor=16 dim=1
    
    for (int k = 0; k < MAX_K; ++k) {
        #pragma HLS PIPELINE II=1
        weight_t diff = S[k] - S_min;
        weight_t neg = weight_t(-1) * diff * weight_t(1.0f / p_lambda);
        
        if (neg > weight_t(20.0f)) neg = weight_t(20.0f);
        if (neg < weight_t(-20.0f)) neg = weight_t(-20.0f);
        
        exp_values[k] = hls::expf(neg);
    }
    
    // Sum (HLS will pipeline the loop-carried dependency)
    weight_t sum_all = 0;
    for (int k = 0; k < MAX_K; ++k) {
        #pragma HLS PIPELINE II=5  // ← HLS achieves ~II=1-2 despite dependency
        sum_all += exp_values[k];
    }
    
    // Normalize (parallel)
    weight_t inv_eta = (sum_all > weight_t(1e-6f)) 
        ? weight_t(1.0f / sum_all) 
        : weight_t(1e6f);
    
    for (int k = 0; k < MAX_K; ++k) {
        #pragma HLS PIPELINE II=1
        W[k] = exp_values[k] * inv_eta;
    }
}
