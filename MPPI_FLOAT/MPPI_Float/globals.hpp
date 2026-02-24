#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include "datatypes.hpp"
#include "hls_math.h"
#include "hls_stream.h"
#include "ref_path.hpp"
#include <iostream>
#include <cmath>


//------------------------------------------------------------------------------
// Compile‐time constants
//------------------------------------------------------------------------------
constexpr int MAX_K  = 2048;
constexpr int MAX_T  = 32;
constexpr int dim_u  = 2;
constexpr int dim_x  = 4;

// Vehicle physical constant (radians)
static const position_s wheel_base = position_s(2.5f);

//------------------------------------------------------------------------------
// Runtime‐configurable globals
//------------------------------------------------------------------------------

// Actual trajectory count and horizon
extern int K_runtime;   // ≤ MAX_K
extern int T_runtime;   // ≤ MAX_T

// MPPI parameters (Q16.16 fixed‐point)
extern float param_exploration;
extern float   param_lambda;
extern float param_alpha;
extern float   param_gamma;

// Control limits
static const fix_t max_steer_abs  = fix_t(0.523f);
static const fix_t max_accel_abs = fix_t(2.0f);

// Cost weights
extern float stage_cost_weight[dim_x];
extern float terminal_cost_weight[dim_x];

// State tracking
extern control_t u_prev[MAX_T][dim_u];
// extern int       prev_waypoints_idx;

// // ============================================================================
// // PACKET STRUCTURE: Groups [k][t] elements atomically
// // ============================================================================
// struct KTStateControlPacket {
//     int k;                      // Trajectory index
//     int t;                      // Time step
//     pos_nn x[dim_x];            // simulated_states[k][t][0:dim_x]
//     control_t v[dim_u];         // v_out[k][t][0:dim_u]
// };

// struct NoisePacket { sample_gauss steer; sample_gauss accel; };
// struct RefPacket  { int t; pos_nn ref[dim_x]; };


#endif // GLOBALS_HPP
