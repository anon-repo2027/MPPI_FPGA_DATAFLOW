
// #endif

#ifndef DATATYPES_HPP
#define DATATYPES_HPP

#include <ap_fixed.h>


/*Float Tests*/

typedef float control_t;     // ±32768 range, 0.000015 resolution
// typedef float sample_t;     // ±32768 range, 0.000015 resolution
// // typedef float cost_t;       // 0-65536 range, 0.000015 resolution
// typedef float exp_t;    
// // ✅ Optimal DSP48E utilization types  
typedef float dsp_optimal_t;    // Perfect for DSP48E slices


/* Datatypes for nearest-neighbor */
// Position: ±32 range, ~0.000061 resolution (16 fractional bits)
typedef float pos_nn;    // 6 integer bits, 16 fractional bits

// Distance squared: 0-2048 range, 16 fractional bits
typedef float cost_nn;       // 12 integer bits, 16 fractional bits

/*Gaussian Noise*/
typedef float sample_gauss; 


/*State Update Datatypes*/
typedef float  position_s;    // ±32 range, 0.00003 resolution (5 int bits for x,y,v up to ~30)
typedef float  velocity_s;    // ±32 range, 0.00003 resolution
typedef float  angle_s;       // ±32 range (angles ±π), 0.00003 resolution
typedef float  control_s;     // ±32 range, 0.00003 resolution
typedef float  dtime_t;       // ±32 range (dt=0.05)

/*Limit Control*/
typedef float fix_t;

/*Filter*/

typedef float high_precision_t;
typedef float  sample_f;      // [−16, 15] with 12 frac bits (CRITICAL PATH - minimize!)


/*Compute Weights*/

// ===== INPUT TYPES =====
typedef float cost_w;        // Can hold up to 102,400

// ===== OUTPUT TYPE ONLY =====
typedef float sample_w;      // 0-4 range, output weights only

typedef float weight_t;    // Output normalized weights
#endif

