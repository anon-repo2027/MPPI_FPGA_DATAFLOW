#include "globals.hpp"

cost_nn calculate_stage_cost_fixed(pos_nn x_t[dim_x], pos_nn ref_point[dim_x], bool is_terminal_cost) {
    #pragma HLS inline
    #pragma HLS ARRAY_PARTITION variable=x_t complete dim=1 
    // read the states
    pos_nn x = x_t[0];
    pos_nn y = x_t[1];
    angle_s yaw = x_t[2];
    velocity_s v = x_t[3];

    // Hardware-friendly yaw normalization
    angle_s temp_yaw = yaw;
    const angle_s TWO_PI = 2.0 * M_PI;
    if (temp_yaw < 0.0) {
        temp_yaw += TWO_PI;
    } else if (temp_yaw >= TWO_PI) {
        temp_yaw -= TWO_PI;
    }
    yaw = temp_yaw;

    pos_nn ref_x = ref_point[0];
    pos_nn ref_y = ref_point[1];
    angle_s ref_yaw = ref_point[2];
    velocity_s ref_v = ref_point[3];

    // Calculate cost using appropriate weights
    float dx = x - ref_x;
    float dy = y - ref_y;
    float dyaw = yaw - ref_yaw;
    float dv = v - ref_v;

    cost_nn cost;
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
