#include "globals.hpp"


void nearest_waypoint_fixed(pos_nn x, pos_nn y, bool update_prev_idx, pos_nn output[dim_x], pos_nn ref_path[1201][4], int &prev_idx_inout) {
#pragma HLS inline
    int search_idx_len = 256;
    int prev_idx = prev_idx_inout;

    cost_nn min_d = 4096.0f;  // Increased to 4096 to be safe
    int nearest_idx = 0;

    FIND_NEAREST_LOOP: for (int i = 0; i < search_idx_len; i++) {
        #pragma HLS PIPELINE II=4
        #pragma HLS LOOP_TRIPCOUNT min=256 max=256
        
        int temp_idx = prev_idx + i;
        int current_idx = (temp_idx >= 1201) ? temp_idx - 1201 : temp_idx;
        
        // Load from BRAM - already in fixed-point
        pos_nn ref_x = ref_path[current_idx][0];
        pos_nn ref_y = ref_path[current_idx][1];

        // All arithmetic in fixed-point
        pos_nn dx = x - ref_x;
        pos_nn dy = y - ref_y;
        cost_nn d_sq = (cost_nn)(dx * dx) + (cost_nn)(dy * dy);


        if (d_sq < min_d) {
            min_d = d_sq;
            nearest_idx = current_idx;

        }
    }

    // Output - already in fixed-point from BRAM
    output[0] = ref_path[nearest_idx][0];
    output[1] = ref_path[nearest_idx][1];
    output[2] = ref_path[nearest_idx][2];
    output[3] = ref_path[nearest_idx][3];

    if (update_prev_idx) {
        prev_idx_inout = nearest_idx;
    }
    // prev_idx_out = prev_waypoints_idx;
}

