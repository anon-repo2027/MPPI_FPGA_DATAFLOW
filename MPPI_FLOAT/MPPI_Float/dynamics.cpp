#include "globals.hpp"

// // Helper to compute trigonometric terms in its own pipeline stage
// void compute_trig(
//     angle_s theta,
//     control_s  steer,
//     angle_s& cos_theta,
//     angle_s& sin_theta,
//     angle_s& tan_steer
// ) {
//     #pragma HLS PIPELINE II=1
//     // Compute sine, cosine, and tangent separately to shorten downstream logic
//     cos_theta  = hls::cos(theta);
//     sin_theta  = hls::sin(theta);
//     tan_steer  = hls::tan(steer);
// }

// State update pulled out into its own pipelined function
void state_update_fixed(
    pos_nn x_t[dim_x],
    pos_nn  v_t[dim_u],
    pos_nn x_t_new[dim_x],
    dtime_t      dt
) {
    #pragma HLS inline
    // #pragma HLS PIPELINE II=45
    #pragma HLS ARRAY_PARTITION variable=x_t     complete dim=1
    #pragma HLS ARRAY_PARTITION variable=v_t     complete dim=1
    #pragma HLS ARRAY_PARTITION variable=x_t_new complete dim=1

    // Local copies for clarity and partitioned accesses
    pos_nn x0 = x_t[0], x1 = x_t[1];
    pos_nn x2 = x_t[2], x3 = x_t[3];
    pos_nn  v0 = v_t[0], v1 = v_t[1];

    // Call compute_trig to isolate trig logic
    angle_s cos_theta, sin_theta, tan_steer;
    // compute_trig(x2, v0, cos_theta, sin_theta, tan_steer);
    cos_theta  = hls::cosf(x2);
    sin_theta  = hls::sinf(x2);
    tan_steer  = hls::tanf(v0);

    // Compute derivatives
    velocity_s dx_dt   = x3 * cos_theta;
    velocity_s dy_dt   = x3 * sin_theta;
    angle_s    dyaw_dt = (x3 / wheel_base) * tan_steer;
    velocity_s dv_dt   = v1;

    // Integrate to get new state
    // dtime_t dt_reg = dtime_t(dt);
    x_t_new[0] = x0 + dx_dt   * dt;
    x_t_new[1] = x1 + dy_dt   * dt;
    x_t_new[2] = x2 + dyaw_dt * dt;
    x_t_new[3] = x3 + dv_dt   * dt;
}