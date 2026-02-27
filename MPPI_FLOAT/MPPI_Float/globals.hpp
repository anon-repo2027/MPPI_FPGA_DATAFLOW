#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include "hls_math.h"
#include "hls_stream.h"
#include "ref_path.hpp"
#include <iostream>
#include <cmath>


//------------------------------------------------------------------------------
// Compile‐time constants
//------------------------------------------------------------------------------
constexpr int MAX_K  = 512;
constexpr int MAX_T  = 16;
constexpr int dim_u  = 2;
constexpr int dim_x  = 4;

// Vehicle physical constant (radians)
static const float wheel_base = (2.5f);

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
static const float max_steer_abs  = (0.523f);
static const float max_accel_abs = (2.0f);

// Cost weights
extern float stage_cost_weight[dim_x];
extern float terminal_cost_weight[dim_x];

// State tracking
extern float u_prev[MAX_T][dim_u];




#endif // GLOBALS_HPP
