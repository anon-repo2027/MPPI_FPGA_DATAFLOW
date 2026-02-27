#include "globals.hpp"

void limit_fixed_safe(float v[dim_u], float limit[dim_u]) {
#pragma HLS inline off

    float steer = v[0];
    float accel = v[1];

    float steer_lim, accel_lim;

    // Clamp steering input
    if (steer >  max_steer_abs) {
        steer_lim =  max_steer_abs;
    } else if (steer < - max_steer_abs) {
        steer_lim = - max_steer_abs;
    } else {
        steer_lim = steer;
    }

    limit[0] =steer_lim;

    // Clamp acceleration input
    if (accel>  max_accel_abs) {
        accel_lim =  max_accel_abs;
    } else if (accel < - max_accel_abs) {
        accel_lim = - max_accel_abs;
    } else {
        accel_lim = accel;
    }

    limit[1] = accel_lim;
}