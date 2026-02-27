#include "globals.hpp"

float calculate_stage_cost_fixed(float x_t[dim_x], float ref_point[dim_x], bool is_terminal_cost) {
    #pragma HLS inline
    #pragma HLS ARRAY_PARTITION variable=x_t complete dim=1 
    // read the states
    float x = x_t[0];
    float y = x_t[1];
    float yaw = x_t[2];
    float v = x_t[3];

    // Hardware-friendly yaw normalization
    float temp_yaw = yaw;
    const float TWO_PI = 2.0 * M_PI;
    if (temp_yaw < 0.0) {
        temp_yaw += TWO_PI;
    } else if (temp_yaw >= TWO_PI) {
        temp_yaw -= TWO_PI;
    }
    yaw = temp_yaw;

    float ref_x = ref_point[0];
    float ref_y = ref_point[1];
    float ref_yaw = ref_point[2];
    float ref_v = ref_point[3];

    // Calculate cost using appropriate weights
    float dx = x - ref_x;
    float dy = y - ref_y;
    float dyaw = yaw - ref_yaw;
    float dv = v - ref_v;

    float cost;
    if (is_terminal_cost) {
        cost = terminal_cost_weight[0] * dx * dx +
               terminal_cost_weight[1] * dy * dy +
               terminal_cost_weight[2] * dyaw * dyaw +
               terminal_cost_weight[3] * dv * dv;
    } else {
        cost = stage_cost_weight[0] * dx * dx +
               stage_cost_weight[1] * dy * dy +
               stage_cost_weight[2] * dyaw * dyaw +
               stage_cost_weight[3] * dv * dv;
    }
    return cost;
}
