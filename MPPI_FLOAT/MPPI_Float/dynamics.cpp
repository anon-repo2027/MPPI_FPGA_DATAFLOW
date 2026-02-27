#include "globals.hpp"


// State update pulled out into its own pipelined function
void state_update_fixed(
    float x_t[dim_x],
    float  v_t[dim_u],
    float x_t_new[dim_x],
    float      dt
) {
    #pragma HLS inline
    // #pragma HLS PIPELINE II=45
    #pragma HLS ARRAY_PARTITION variable=x_t     complete dim=1
    #pragma HLS ARRAY_PARTITION variable=v_t     complete dim=1
    #pragma HLS ARRAY_PARTITION variable=x_t_new complete dim=1

    // Local copies for clarity and partitioned accesses
    float x0 = x_t[0], x1 = x_t[1];
    float x2 = x_t[2], x3 = x_t[3];
    float  v0 = v_t[0], v1 = v_t[1];

    // Call compute_trig to isolate trig logic
    float cos_theta, sin_theta, tan_steer;
    // compute_trig(x2, v0, cos_theta, sin_theta, tan_steer);
    cos_theta  = hls::cosf(x2);
    sin_theta  = hls::sinf(x2);
    tan_steer  = hls::tanf(v0);

    // Compute derivatives
    float dx_dt   = x3 * cos_theta;
    float dy_dt   = x3 * sin_theta;
    float    dyaw_dt = (x3 / wheel_base) * tan_steer;
    float dv_dt   = v1;

    // Integrate to get new state
    // float dt_reg = float(dt);
    x_t_new[0] = x0 + dx_dt   * dt;
    x_t_new[1] = x1 + dy_dt   * dt;
    x_t_new[2] = x2 + dyaw_dt * dt;
    x_t_new[3] = x3 + dv_dt   * dt;
}