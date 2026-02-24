#include "globals.hpp"

// ✅ HIGH-QUALITY XORSHIFT: Tested shift patterns
void generategauss_fixed_point(sample_gauss steer_samples[MAX_K*MAX_T], sample_gauss accel_samples[MAX_K*MAX_T]) {
    // #pragma HLS INLINE
#pragma HLS inline
    ap_uint<32> seed1 = 0x9E3779B9;
    ap_uint<32> seed2 = 0x85EBCA6B;

    const float TWO_PI = 6.28318530717958647692f;
    const float MIN_U  = 1e-9f;

GENERATE_LOOP:
    for (int idx = 0; idx < MAX_K * MAX_T; idx++) {
        #pragma HLS PIPELINE II=1

        // XorShift PRNG
        seed1 ^= seed1 >> 13; 
        seed1 ^= seed1 << 17; 
        seed1 ^= seed1 >> 5; 
        seed1 += 0x9E3779B9;

        seed2 ^= seed2 >> 13; 
        seed2 ^= seed2 << 17; 
        seed2 ^= seed2 >> 5; 
        seed2 += 0x85EBCA6B;

        // Extract 24-bit random values (high quality bits)
        uint32_t u1_raw = (seed1.to_uint() >> 8) & 0xFFFFFF;
        uint32_t u2_raw = (seed2.to_uint() >> 8) & 0xFFFFFF;

        // Convert to float [0,1) - NO FIXED POINT YET
        float u1 = float(u1_raw) * (1.0f / 16777216.0f); // 1/2^24
        float u2 = float(u2_raw) * (1.0f / 16777216.0f);

        // Safety clamp
        if (u1 < MIN_U) u1 = MIN_U;

        // Box-Muller in float
        float radius = hls::sqrtf(-2.0f * hls::logf(u1));
        float theta = TWO_PI * u2;

        float sin_val = hls::sinf(theta);        
        float cos_val = hls::cosf(theta);

        // ONLY NOW cast to fixed-point for storage
        steer_samples[idx] = sample_gauss(radius * sin_val);
        accel_samples[idx] = sample_gauss(radius * cos_val);
    }

}