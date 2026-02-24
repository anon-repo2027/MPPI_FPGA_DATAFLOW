#include "globals.hpp"

void limit_fixed_safe(pos_nn v[dim_u], pos_nn limit[dim_u]) {
#pragma HLS inline off
// #pragma HLS INTERFACE bram storage_type=ram_1p port=v depth=2
// #pragma HLS INTERFACE bram storage_type=ram_1p port=limit depth=2
// #pragma HLS INTERFACE s_axilite port=return bundle=control

    pos_nn steer = v[0];
    pos_nn accel = v[1];

    pos_nn steer_lim, accel_lim;

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