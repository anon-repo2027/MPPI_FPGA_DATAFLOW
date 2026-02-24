#include "globals.hpp"

// Calculate determinant of 2×2 matrix [a b; c d] stored as [a, b, c, d]
dsp_optimal_t determinant2x2_fixed(dsp_optimal_t matrix[4]) {
    #pragma HLS inline
    // det = a*d – b*c
    return matrix[0] * matrix[3] - matrix[1] * matrix[2];
}

// Compute inverse of 2×2 matrix: result = (1/det) * [ d, –b; –c, a ]
void inverse2x2_fixed(dsp_optimal_t matrix[4], dsp_optimal_t result[4]) {
    #pragma HLS inline
    dsp_optimal_t det    = determinant2x2_fixed(matrix);
    dsp_optimal_t invDet = dsp_optimal_t(1.0f) / det;

    // Unroll to compute all four entries concurrently
    // #pragma HLS UNROLL factor=4
    result[0] =  matrix[3] * invDet;  // [0,0]
    result[1] = -matrix[1] * invDet;  // [0,1]
    result[2] = -matrix[2] * invDet;  // [1,0]
    result[3] =  matrix[0] * invDet;  // [1,1]
}

// Matrix-vector multiply: result = matrix * vec
// matrix as [m00,m01,m10,m11], vec as [v0,v1]
void matVecMult_fixed(dsp_optimal_t matrix[4], dsp_optimal_t vec[2], dsp_optimal_t result[2]) {
    #pragma HLS inline
    // Pipeline each output element
    MATVEC_LOOP: for (int i = 0; i < 2; ++i) {
        #pragma HLS PIPELINE II=1
        dsp_optimal_t acc = dsp_optimal_t(0.0f);
        // Unroll inner loop
        
        for (int j = 0; j < 2; ++j) {
            #pragma HLS UNROLL factor=2
            acc += matrix[i*2 + j] * vec[j];
        }
        result[i] = acc;
    }
}

// Dot product of two 2-element vectors
dsp_optimal_t dotProduct_fixed(dsp_optimal_t vec1[2], dsp_optimal_t vec2[2]) {
    #pragma HLS inline
    dsp_optimal_t sum = dsp_optimal_t(0.0f);
    // Unroll both iterations for parallel multiply-add
    
    for (int i = 0; i < 2; ++i) {
        #pragma HLS UNROLL factor=2
        sum += vec1[i] * vec2[i];
    }
    return sum;
}