#include "globals.hpp"


void compute_weight_fixed(float S[MAX_K], float W[MAX_K], float p_lambda) {
    #pragma HLS inline
    
    #pragma HLS ARRAY_PARTITION variable=S cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=W cyclic factor=16 dim=1
    
    // Find min using a parallel reduction tree to shorten the critical path
    float min_vals[MAX_K];
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
            float val1 = min_vals[i];
            float val2 = min_vals[i + d];
            if (val2 < val1) {
                min_vals[i] = val2;
            }
        }
    }
    float S_min = min_vals[0];
    
    // Compute exponentials (parallel)
    static float exp_values[MAX_K];
    #pragma HLS ARRAY_PARTITION variable=exp_values cyclic factor=16 dim=1
    
    for (int k = 0; k < MAX_K; ++k) {
        #pragma HLS PIPELINE II=1
        float diff = S[k] - S_min;
        float neg = float(-1) * diff * float(1.0f / p_lambda);
        
        if (neg > float(20.0f)) neg = float(20.0f);
        if (neg < float(-20.0f)) neg = float(-20.0f);
        
        exp_values[k] = hls::expf(neg);
    }
    
    // Sum (HLS will pipeline the loop-carried dependency)
    float sum_all = 0;
    for (int k = 0; k < MAX_K; ++k) {
        #pragma HLS PIPELINE II=5  // ← HLS achieves ~II=1-2 despite dependency
        sum_all += exp_values[k];
    }
    
    // Normalize (parallel)
    float inv_eta = (sum_all > float(1e-6f)) 
        ? float(1.0f / sum_all) 
        : float(1e6f);
    
    for (int k = 0; k < MAX_K; ++k) {
        #pragma HLS PIPELINE II=1
        W[k] = exp_values[k] * inv_eta;
    }
}
