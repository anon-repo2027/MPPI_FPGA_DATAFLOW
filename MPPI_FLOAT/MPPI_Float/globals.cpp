#include "globals.hpp"

float param_exploration = (0.1f);
float   param_lambda      = (100.0f);
float param_alpha       = (0.98f);
float   param_gamma       = (0.0f);

float stage_cost_weight[dim_x]    = {(50.0f), (50.0f), (1.0f),  (20.0f)};
float terminal_cost_weight[dim_x] = {(50.0f), (50.0f), (1.0f),  (20.0f)};

float u_prev[MAX_T][dim_u] = {};

