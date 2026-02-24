#include "datatypes.hpp"
#include "globals.hpp"


// ========== STREAM DATA STRUCTURES ==========
// struct TrajectoryState {
//     int k;
//     int t;
//     pos_nn x[dim_x];
//     control_t v[dim_u];
// };


// // ✅ NEW: Define a bundle to hold all T states for one trajectory (k).
// struct TrajectoryBundle {
//     TrajectoryState states[MAX_T];
// };

struct CostOutput {
    int k;
    cost_w total_cost;
};


// ✅ Compact state (no k, t - they're implicit in loops)
struct CompactTrajectoryState {
    pos_nn x[dim_x];      // 128 bits (4 floats)
    pos_nn v[dim_u];      // 64 bits (2 floats)
    // Total: 192 bits per state
};

// ✅ Pack 2 compact states per write
struct TrajectoryPair {
    CompactTrajectoryState s[2];
    // Total: 384 bits per pair ✓
};

// ========== FORWARD DECLARATIONS (STREAMING) ==========
void initialize_mppi_fixed(float observed_x[dim_x], dsp_optimal_t sigma_inv[dim_x], control_t u[MAX_T][dim_u], pos_nn x0[dim_x]);
void generategauss_fixed_point(sample_gauss steer_samples[MAX_K*MAX_T], sample_gauss accel_samples[MAX_K*MAX_T]);
void precalc_nominal_refs_fixed(pos_nn x0[dim_x], control_t u[MAX_T][dim_u], pos_nn ref_path_f32[1201][4], pos_nn nominal_ref_points[MAX_T][dim_x], dtime_t dt, int &nearest_wp_idx);

void simulate_trajectories_stream(
    pos_nn x0[dim_x], 
    control_t u[MAX_T][dim_u], 
    sample_gauss steer_samples[MAX_K*MAX_T], 
    sample_gauss accel_samples[MAX_K*MAX_T], 
    dtime_t dt,
    float p_expl, // Pass parameter
    hls::stream<TrajectoryPair>& trajectory_out
);

void calculate_all_costs_stream(
    hls::stream<TrajectoryPair>& trajectory_in, 
    pos_nn nominal_ref_points[MAX_T][dim_x], 
    control_t u[MAX_T][dim_u], 
    dsp_optimal_t sigma_inv[dim_x], 
    float p_gamma, // Pass parameter
    hls::stream<CostOutput>& cost_out
);

// void update_control_sequence_stream(
//     hls::stream<CostOutput>& cost_in, 
//     sample_gauss steer_samples[MAX_K*MAX_T], 
//     sample_gauss accel_samples[MAX_K*MAX_T], 
//     control_t u[MAX_T][dim_u], 
//     control_t u_out[dim_u]
// );
void compute_weights_stream(
    hls::stream<CostOutput>& cost_in,
    float p_lambda, // Pass parameter
    hls::stream<sample_w>& weight_out
);

// void accumulate_weighted_noise_stream(
//     hls::stream<sample_w>& weight_in,
//     sample_gauss steer_samples[MAX_K*MAX_T],
//     sample_gauss accel_samples[MAX_K*MAX_T],
//     control_t u[MAX_T][dim_u],
//     control_t u_out[dim_u]
// );

void accumulate_weighted_noise_stream(
    hls::stream<sample_w>& weight_in,
    sample_gauss steer_samples[MAX_K*MAX_T],
    sample_gauss accel_samples[MAX_K*MAX_T],
    control_t u_read[MAX_T][dim_u],    // ← Input: what to read from
    control_t u_write[MAX_T][dim_u],   // ← Output: what to write to (NEW)
    control_t u_out[dim_u]
);

// ========== HELPER PROTOTYPES (UNCHANGED) ==========
void nearest_waypoint_fixed(pos_nn x, pos_nn y, bool update_prev_idx, pos_nn output[dim_x], pos_nn ref_path[1201][4], int &prev_idx_out);
cost_nn calculate_stage_cost_fixed(pos_nn x_t[dim_x], pos_nn ref_point[dim_x], bool is_terminal_cost);
void state_update_fixed(pos_nn x[dim_x], pos_nn g[dim_u], pos_nn result[dim_x], dtime_t dt);
void limit_fixed_safe(pos_nn v[dim_u], pos_nn result[dim_u]);
void compute_weight_fixed(cost_w S[MAX_K], sample_w W[MAX_K], float p_lambda);
void average_fixed(sample_f xx[MAX_T*dim_u], sample_f xx_avg[MAX_T*dim_u], int window_size);
void inverse2x2_fixed(dsp_optimal_t matrix[dim_x], dsp_optimal_t result[dim_x]);
dsp_optimal_t determinant2x2_fixed(dsp_optimal_t matrix[4]);
void matVecMult_fixed(dsp_optimal_t matrix[4], dsp_optimal_t vec[2], dsp_optimal_t result[2]);
dsp_optimal_t dotProduct_fixed(dsp_optimal_t vec1[2], dsp_optimal_t vec2[2]);

// // ========== GLOBALS (UNCHANGED) ==========
// static pos_nn ref_path_f32[1201][4];
// static bool ref_path_initialized = false;

// void convert_ref_path_once(float ref_path_array[4804]) {
//     if (!ref_path_initialized) {
//         for (int i = 0; i < 1201; i++) {
//             #pragma HLS PIPELINE II=1
//             for (int j = 0; j < 4; j++) {
//                 #pragma HLS UNROLL
//                 ref_path_f32[i][j] = pos_nn(ref_path_array[i * 4 + j]);
//             }
//         }
//         ref_path_initialized = true;
//     }
// }
// Static arrays for data persistence and sharing
static sample_gauss steer_samples[MAX_K*MAX_T];
static sample_gauss accel_samples[MAX_K*MAX_T];
static pos_nn nominal_ref_points[MAX_T][dim_x];
static int nearest_wp_idx = 0;

// Read buffer: used by Stages 3-4
static control_t u_read[MAX_T][dim_u];


// Write buffer: output from Stage 5B
static control_t u_write[MAX_T][dim_u];


// ✅ NEW: Static buffers for intermediate results (avoid stack allocation)
static control_t w_epsilon[MAX_T][dim_u];
static sample_f w_epsilon_flat[MAX_T*dim_u];
static sample_f w_epsilon_filtered[MAX_T*dim_u];
static cost_w S[MAX_K];
static sample_w W[MAX_K];


// ✅ NEW: Wrapper function to handle parameter updates canonically in dataflow
void update_mppi_parameters(float p_expl, float p_lambda, float p_alpha, float s_w[dim_x], float t_w[dim_x]) {
    #pragma HLS inline
    param_exploration = p_expl;
    param_lambda = p_lambda;
    param_alpha = p_alpha;
    param_gamma = p_lambda * (1.0f - p_alpha);
    for (int i = 0; i < dim_x; i++) {
        #pragma HLS UNROLL
        stage_cost_weight[i] = s_w[i];
        terminal_cost_weight[i] = t_w[i];
    }
}

// ✅ NEW: Wrapper function to update u_prev buffer
void update_u_prev_buffer(control_t u_write[MAX_T][dim_u]) {
    #pragma HLS INLINE
    for (int t = 0; t < MAX_T - 1; t++) {
        #pragma HLS UNROLL
        for (int i = 0; i < dim_u; i++) {
            u_prev[t][i] = u_write[t + 1][i];
        }
    }
    for (int i = 0; i < dim_u; i++) {
        #pragma HLS UNROLL
        u_prev[MAX_T - 1][i] = u_write[MAX_T - 1][i];
    }
}


/*************************************************************************************/
/* Top-Level Function: Streaming Version                                             */
/*************************************************************************************/
void calc_control_input(float observed_x[dim_x], float ref_path_array[1201][4], float u_out[dim_u], float all_params[12]) {
 
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE bram port=observed_x storage_type=ram_1p
    #pragma HLS INTERFACE bram port=ref_path_array 
    #pragma HLS INTERFACE bram port=u_out storage_type=ram_1p
    #pragma HLS INTERFACE bram port=all_params storage_type=ram_1p
    // #pragma HLS INTERFACE s_axilite port=nearest_wp_idx_inout bundle=control

    // ✅ Partition the input array to enable 4 parallel reads
    // #pragma HLS ARRAY_PARTITION variable=ref_path_array complete dim=1
    
    dtime_t dt = dtime_t(all_params[0]);
    float exploration_local = (all_params[1]);
    float lambda_local = (all_params[2]);
    float alpha_local = (all_params[3]);
    
    float stage_weights_local[dim_x];
    float terminal_weights_local[dim_x];
    for (int i = 0; i < dim_x; i++) {
        #pragma HLS UNROLL
        stage_weights_local[i] = (all_params[4 + i]);
        terminal_weights_local[i] = (all_params[8 + i]);
    }

    // convert_ref_path_once(ref_path_array);

    // #pragma HLS BIND_STORAGE variable=nominal_ref_points type=RAM_2P impl=bram
    // #pragma HLS ARRAY_PARTITION variable=steer_samples cyclic factor=16 dim=1
    // #pragma HLS ARRAY_PARTITION variable=accel_samples cyclic factor=16 dim=1
    // #pragma HLS BIND_STORAGE variable=u_read type=RAM_2P impl=bram
    // #pragma HLS BIND_STORAGE variable=u_write type=RAM_2P impl=bram

    #pragma HLS BIND_STORAGE variable=nominal_ref_points type=RAM_2P impl=bram
    #pragma HLS BIND_STORAGE variable=steer_samples type=RAM_2P impl=bram
    #pragma HLS BIND_STORAGE variable=accel_samples type=RAM_2P impl=bram
    #pragma HLS BIND_STORAGE variable=u_read type=RAM_2P impl=bram
    #pragma HLS BIND_STORAGE variable=u_write type=RAM_2P impl=bram
    #pragma HLS BIND_STORAGE variable=w_epsilon type=RAM_2P impl=bram
    #pragma HLS BIND_STORAGE variable=S type=RAM_2P impl=bram
    #pragma HLS BIND_STORAGE variable=W type=RAM_2P impl=bram
    
    #pragma HLS ARRAY_PARTITION variable=steer_samples cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=accel_samples cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=w_epsilon complete dim=2
    #pragma HLS ARRAY_PARTITION variable=S cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=W cyclic factor=16 dim=1
    // Local variables for dataflow stages
    pos_nn x0[dim_x];
    control_t u[MAX_T][dim_u];
    dsp_optimal_t sigma_inv[dim_x];
    control_t u_internal[dim_u];
    
    // int start_idx = nearest_wp_idx_inout;
    
    // // Streams for connecting dataflow stages
    // hls::stream<TrajectoryBundle> trajectory_stream("trajectory_stream");
    // hls::stream<CostOutput> cost_stream("cost_stream");
    // #pragma HLS STREAM variable=trajectory_stream depth=128
    // #pragma HLS STREAM variable=cost_stream depth=128
    // Streams for connecting dataflow stages

    // Use two 'u' arrays to break the read/write dependency in the dataflow
    // control_t u_current[MAX_T][dim_u];
    // control_t u_next[MAX_T][dim_u];
    // static int nearest_wp_idx = 0;


    update_mppi_parameters(exploration_local, lambda_local, alpha_local, stage_weights_local, terminal_weights_local);
    // ✅ CORRECT: Generate all noise BEFORE the dataflow region.
    // This ensures the noise is static and identical to your original code.
    generategauss_fixed_point(steer_samples, accel_samples);
 
    // STAGE 2: Calculate the constant inputs for the pipeline
    initialize_mppi_fixed(observed_x, sigma_inv, u_read, x0);
    // initialize_mppi_fixed(observed_x, sigma_inv, u_next, nullptr); // Initialize u_next, x0 output is ignored

    

    
    
    #pragma HLS DATAFLOW

    // // Update global parameters (must be inside dataflow if used by its functions)
    // param_exploration = exploration_local;
    // param_lambda = lambda_local;
    // param_alpha = alpha_local;
    // param_gamma = lambda_local * (1.0f - alpha_local);
    // for (int i = 0; i < dim_x; i++) {
    //     #pragma HLS UNROLL
    //     stage_cost_weight[i] = stage_weights_local[i];
    //     terminal_cost_weight[i] = terminal_weights_local[i];
    // }

    // // STAGE 1: Initialize
    // initialize_mppi_fixed(observed_x, sigma_inv, u, x0);
    
    // // STAGE 2: Pre-calculate References (writes to BRAM)
    // precalc_nominal_refs_fixed(x0, u, ref_path_f32, nominal_ref_points, dt, nearest_wp_idx);
    
    // // STAGE 3: Simulate Trajectories (reads noise array, writes to stream)
    // simulate_trajectories_stream(x0, u, steer_samples, accel_samples, dt, trajectory_stream);
    
    // // STAGE 4: Calculate Costs (reads ref BRAM, consumes trajectory stream, writes cost stream)
    // calculate_all_costs_stream(trajectory_stream, nominal_ref_points, u, sigma_inv, cost_stream);
    
    // // STAGE 5: Update Control (consumes cost stream, reads noise array)
    // update_control_sequence_stream(cost_stream, steer_samples, accel_samples, u, u_internal);
    // // STAGE 1: Initialize. Writes to u_current and u_next.
    // initialize_mppi_fixed(observed_x, sigma_inv, u_current, x0);
    // // Also initialize u_next to ensure it has the base values before update
    // initialize_mppi_fixed(observed_x, sigma_inv, u_next, x0);
    hls::stream<TrajectoryPair> trajectory_stream("trajectory_stream");
    hls::stream<CostOutput> cost_stream("cost_stream");
    hls::stream<sample_w> weight_stream("weight_stream");  // ✅ NEW stream
    
    #pragma HLS STREAM variable=trajectory_stream depth=16384
    #pragma HLS STREAM variable=cost_stream depth=2048
    #pragma HLS STREAM variable=weight_stream depth=2048    // ✅ NEW stream depth

    // STAGE 3: Pre-calculate References. Reads from u_current.
    precalc_nominal_refs_fixed(x0, u_read, ref_path_array, nominal_ref_points, dt, nearest_wp_idx);
    
    // STAGE 4: Simulate Trajectories. Reads from u_current.
    simulate_trajectories_stream(x0, u_read, steer_samples, accel_samples, dt, exploration_local, trajectory_stream);
    
    // STAGE 5: Calculate Costs. Reads from u_current.
    calculate_all_costs_stream(trajectory_stream, nominal_ref_points, u_read, sigma_inv, (lambda_local * (1.0f - alpha_local)), cost_stream);
    
    // // STAGE 5: Update Control. Reads u_current, writes to u_next.
    // update_control_sequence_stream(cost_stream, steer_samples, accel_samples, u_current, u_internal);

    // STAGE 6A: Compute Weights (starts as soon as costs arrive)
    compute_weights_stream(cost_stream, lambda_local, weight_stream);
    
    // STAGE 6B: Accumulate Weighted Noise (starts as soon as weights arrive)
    accumulate_weighted_noise_stream(weight_stream, steer_samples, accel_samples, u_read, u_write, u_internal);

    // limit_fixed_safe(u_internal, u_internal);

    // // Final output conversion
    // u_out[0] = float(u_internal[0]);
    // u_out[1] = float(u_internal[1]);
    // STAGE 7: Final control output limiting
    limit_fixed_safe(u_internal, u_out);

    // STAGE 8: Update the control sequence for the next iteration
    update_u_prev_buffer(u_write);
}

/*************************************************************************************/
/* Stage 1: Initialization (Updated to include ref_path conversion)                  */
/*************************************************************************************/
void initialize_mppi_fixed(float observed_x[dim_x], dsp_optimal_t sigma_inv[dim_x], control_t u[MAX_T][dim_u], pos_nn x0[dim_x]) {
    // #pragma HLS PIPELINE II=1
    #pragma HLS inline
    // Fixed-point sigma matrix
    dsp_optimal_t sigma[dim_x] = {
        dsp_optimal_t(0.075f), dsp_optimal_t(0.0f), 
        dsp_optimal_t(0.0f), dsp_optimal_t(2.0f)
    };
    inverse2x2_fixed(sigma, sigma_inv);

    // Initialize control sequence from previous iteration
    for (int i = 0; i < MAX_T; i++) {
        #pragma HLS UNROLL
        for (int j = 0; j < dim_u; j++) {
            u[i][j] = u_prev[i][j];
        }
    }

    // ✅ FIX: Add a null check to prevent segmentation fault.
    // Only write to x0 if the pointer is valid.
    // Convert observed state from float to fixed-point
    for (int i = 0; i < dim_x; i++) {
        #pragma HLS UNROLL
        x0[i] = pos_nn(observed_x[i]);
    }
    
}

/*************************************************************************************/
/* Stage 2: Pre-calculate Nominal Reference Path (Simplified)                        */
/*************************************************************************************/
// void precalc_nominal_refs_fixed(pos_nn x0_in[dim_x], control_t u[MAX_T][dim_u], pos_nn ref_path_array[1201][4], pos_nn nominal_ref_points[MAX_T][dim_x], dtime_t dt, int &start_wp_idx) {
// // #pragma HLS BIND_STORAGE variable=ref_path_f32 type=RAM_1WNR impl=bram
// // #pragma HLS bind_storage variable=ref_path_f32 type=RAM_2P impl=bram
// #pragma HLS ARRAY_PARTITION variable=ref_path_array complete dim=2
// // #pragma HLS PIPELINE II=1
// #pragma HLS inline off
//     pos_nn u_fp[dim_u];
//     pos_nn x0[dim_x];
//     #pragma HLS ARRAY_PARTITION variable=x0 complete dim=1
//     for(int i=0; i<dim_x; i++) {
//         #pragma HLS UNROLL
//         x0[i] = x0_in[i];
//     }
    
//     // int prev_idx_dummy = start_wp_idx;
//     int local_prev_idx = start_wp_idx;
//     pos_nn ref_output[dim_x];
//     pos_nn g_result[dim_u];
//     pos_nn x_new[dim_x];
//     #pragma HLS ARRAY_PARTITION variable=x_new complete dim=1
//     // fix_t u_int[MAX_T][dim_u];
//     // fix_t g_result_int[dim_u];

//     // âœ… Now we can use ref_path_f32 directly - no conversion needed!
//     nearest_waypoint_fixed(x0[0], x0[1], true, nominal_ref_points[0], ref_path_array, start_wp_idx);
//     local_prev_idx = start_wp_idx; // Sync local copy after the update

//     PRE_CALC_REF_T: for (int t = 1; t < MAX_T; t++) {
//         // #pragma HLS LOOP_TRIPCOUNT min=16 max=64 avg=32
//         #pragma HLS PIPELINE II=47
       
//         nearest_waypoint_fixed(x0[0], x0[1], false, ref_output, ref_path_array, local_prev_idx);
//         for(int i=0; i<dim_x; i++) { 
//             #pragma HLS UNROLL
//             nominal_ref_points[t][i] = ref_output[i]; 
//         }
        
//         u_fp[0] = u[t][0];
//         u_fp[1] = u[t][1];
//         limit_fixed_safe(u_fp, g_result);
        
//         state_update_fixed(x0, g_result, x_new, dt);
//         for(int i=0; i<dim_x; i++) { 
//             #pragma HLS UNROLL
//             x0[i] = x_new[i]; 
//         }
//     }
// }

void precalc_nominal_refs_fixed(
    pos_nn x0_in[dim_x], 
    control_t u[MAX_T][dim_u], 
    pos_nn ref_path_array[1201][4], 
    pos_nn nominal_ref_points[MAX_T][dim_x], 
    dtime_t dt, 
    int &start_wp_idx
) {
    // #pragma HLS ARRAY_PARTITION variable=ref_path_array complete dim=2
    #pragma HLS inline off
    #pragma HLS BIND_STORAGE variable=ref_path_array type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=ref_path_array cyclic factor=4 dim=2
    
    pos_nn u_fp[dim_u];
    
    // ✅ TWO buffers instead of one
    pos_nn x_current[dim_x];  // ← Read from this
    pos_nn x_next[dim_x];     // ← Write to this
    #pragma HLS ARRAY_PARTITION variable=x_current complete dim=1
    #pragma HLS ARRAY_PARTITION variable=x_next complete dim=1
    
    for(int i=0; i<dim_x; i++) {
        #pragma HLS UNROLL
        x_current[i] = x0_in[i];
    }
    
    int local_prev_idx = start_wp_idx;
    pos_nn ref_output[dim_x];
    pos_nn g_result[dim_u];
    
    // nearest_waypoint_fixed(x_current[0], x_current[1], true, nominal_ref_points[0], ref_path_array, start_wp_idx);
    // local_prev_idx = start_wp_idx;
    
    // PRE_CALC_REF_T: for (int t = 0; t < MAX_T; t++) {
    //     // #pragma HLS PIPELINE II=1 rewind
        
    //     nearest_waypoint_fixed(x_current[0], x_current[1], false, ref_output, ref_path_array, local_prev_idx);
    //     for(int i=0; i<dim_x; i++) { 
    //         #pragma HLS UNROLL
    //         nominal_ref_points[t][i] = ref_output[i]; 
    //     }
        
    //     u_fp[0] = u[t][0];
    //     u_fp[1] = u[t][1];
    //     limit_fixed_safe(u_fp, g_result);
        
    //     // ✅ Read from x_current, WRITE to x_next (different variables!)
    //     state_update_fixed(x_current, g_result, x_next, dt);
        
    //     // ✅ Copy x_next back to x_current for next iteration
    //     for(int i=0; i<dim_x; i++) { 
    //         #pragma HLS UNROLL
    //         x_current[i] = x_next[i];  // ← Simple register copy, fast
    //     }
    // }
    // ✅ Single loop with initialization handled on first iteration
    PRE_CALC_REF_T: for (int t = 0; t < MAX_T; t++) {
        #pragma HLS PIPELINE II=1 rewind
        
        // ✅ On first iteration (t==0), use update_prev_idx=true
        bool is_first = (t == 0);
        
        nearest_waypoint_fixed(
            x_current[0], x_current[1], 
            is_first,  // Only update prev_idx on first call
            ref_output, 
            ref_path_array, 
            local_prev_idx
        );
        
        for(int i=0; i<dim_x; i++) { 
            #pragma HLS UNROLL
            nominal_ref_points[t][i] = ref_output[i]; 
        }
        
        u_fp[0] = u[t][0];
        u_fp[1] = u[t][1];
        limit_fixed_safe(u_fp, g_result);
        
        state_update_fixed(x_current, g_result, x_next, dt);
        
        for(int i=0; i<dim_x; i++) { 
            #pragma HLS UNROLL
            x_current[i] = x_next[i];
        }
    }
    
    start_wp_idx = local_prev_idx;
}

/*************************************************************************************/
/* STAGE 3: Simulate Trajectories (Streaming Version)                                */
/*************************************************************************************/
// void simulate_trajectories_stream(
//     pos_nn x0_in[dim_x], 
//     control_t u[MAX_T][dim_u], 
//     sample_gauss steer_samples[MAX_K*MAX_T], 
//     sample_gauss accel_samples[MAX_K*MAX_T], 
//     dtime_t dt,
//     hls::stream<TrajectoryState>& trajectory_out
// ) {
//     const float exploration_threshold = (1.0f) - param_exploration;
//     const float k_threshold = exploration_threshold * (MAX_K);
//     pos_nn v_out_f[dim_u];
//     pos_nn x0[dim_x];
    
//     // ✅ USE YOUR ORIGINAL, CORRECT LOOP STRUCTURE
//     TRAJECTORY_SIM_K: for (int k = 0; k < MAX_K; k++) {
//         #pragma HLS UNROLL factor=16
        
//         // pos_nn x_k[dim_x];
//         #pragma HLS ARRAY_PARTITION variable=x0 complete dim=1
        
//         STATE_INIT: for (int i = 0; i < dim_x; i++) { 
//             #pragma HLS UNROLL
//             x0[i] = x0_in[i]; 
//         }

//         TRAJECTORY_SIM_T: for (int t = 0; t < MAX_T; t++) {
//             #pragma HLS PIPELINE II=1
            
//             control_t v_k[dim_u];
//             if ((k) < k_threshold) {
//                 v_k[0] = u[t][0] + steer_samples[k*MAX_T+t];
//                 v_k[1] = u[t][1] + accel_samples[k*MAX_T+t];
//             } else {
//                 v_k[0] = steer_samples[k*MAX_T+t];
//                 v_k[1] = accel_samples[k*MAX_T+t];
//             }

//             pos_nn g_result[dim_u];
//             pos_nn x_new[dim_x];
//             #pragma HLS ARRAY_PARTITION variable=g_result complete dim=1
//             #pragma HLS ARRAY_PARTITION variable=x_new complete dim=1
            
//             v_out_f[0] = v_k[0];
//             v_out_f[1] = v_k[1];
//             limit_fixed_safe(v_out_f, g_result);
//             state_update_fixed(x0, g_result, x_new, dt);
            
//             STATE_UPDATE: for (int i = 0; i < dim_x; i++) { 
//                 #pragma HLS UNROLL
//                 x0[i] = x_new[i]; 
//             }
            
//             // ✅ STREAM WRITE: Pack and send data
//             TrajectoryState packet;
//             packet.k = k;
//             packet.t = t;
//             for (int i = 0; i < dim_x; i++) packet.x[i] = x_new[i];
//             for (int i = 0; i < dim_u; i++) packet.v[i] = v_k[i];
//             trajectory_out.write(packet);
//         }
//     }
// }
// void simulate_trajectories_stream(
//     pos_nn x0_in[dim_x], 
//     control_t u[MAX_T][dim_u], 
//     sample_gauss steer_samples[MAX_K*MAX_T], 
//     sample_gauss accel_samples[MAX_K*MAX_T], 
//     dtime_t dt,
//     hls::stream<TrajectoryPair>& trajectory_out
// ) {
// // #pragma HLS PIPELINE II=1
// #pragma HLS inline off
//     // Create a local, partitioned copy of x0 to remove dependency.
//     pos_nn x0[dim_x];
//     #pragma HLS ARRAY_PARTITION variable=x0 complete dim=1
//     for(int i=0; i<dim_x; i++) {
//         #pragma HLS UNROLL
//         x0[i] = x0_in[i];
//     }

//     const float exploration_threshold = (1.0f) - param_exploration;
//     const float k_threshold = exploration_threshold * (MAX_K);
//     pos_nn v_out_f[dim_u];
    
//     TRAJECTORY_SIM_K: for (int k = 0; k < MAX_K; k++) {
//         // #pragma HLS UNROLL factor=16
//         #pragma HLS PIPELINE II=1 
//         pos_nn x_k[dim_x];
//         #pragma HLS ARRAY_PARTITION variable=x_k complete dim=1
        
//         STATE_INIT: for (int i = 0; i < dim_x; i++) { 
//             #pragma HLS UNROLL
//             x_k[i] = x0[i]; 
//         }
        
//         // ✅ Create a local bundle to be filled.
//         TrajectoryBundle bundle;
//         #pragma HLS ARRAY_PARTITION variable=bundle.states complete dim=1


//         TRAJECTORY_SIM_T: for (int t = 0; t < MAX_T; t++) {
//             // #pragma HLS PIPELINE II=1 // Keep the target II=1
//             #pragma HLS UNROLL factor=2
            
//             control_t v_k[dim_u];
//             if ((k) < k_threshold) {
//                 v_k[0] = u[t][0] + steer_samples[k*MAX_T+t];
//                 v_k[1] = u[t][1] + accel_samples[k*MAX_T+t];
//             } else {
//                 v_k[0] = steer_samples[k*MAX_T+t];
//                 v_k[1] = accel_samples[k*MAX_T+t];
//             }

//             pos_nn g_result[dim_u];
//             #pragma HLS ARRAY_PARTITION variable=g_result complete dim=1
            
//             v_out_f[0] = v_k[0];
//             v_out_f[1] = v_k[1];
//             limit_fixed_safe(v_out_f, g_result);
//             state_update_fixed(x_k, g_result, x_k, dt);
            
//             // // ✅ FIX: Inline the state_update_fixed logic directly here
//             // // This allows HLS to pipeline the long-latency math operations.
//             // pos_nn x_old[dim_x];
//             // #pragma HLS ARRAY_PARTITION variable=x_old complete dim=1
//             // for(int i=0; i<dim_x; i++) 
//             // x_old[i] = x_k[i];

//             // angle_s cos_theta = hls::cosf(x_old[2]);
//             // angle_s sin_theta = hls::sinf(x_old[2]);
//             // angle_s tan_steer = hls::tanf(g_result[0]);

//             // velocity_s dx_dt   = x_old[3] * cos_theta;
//             // velocity_s dy_dt   = x_old[3] * sin_theta;
//             // angle_s    dyaw_dt = (x_old[3] / wheel_base) * tan_steer;
//             // velocity_s dv_dt   = g_result[1];

//             // x_k[0] = x_old[0] + dx_dt   * dt;
//             // x_k[1] = x_old[1] + dy_dt   * dt;
//             // x_k[2] = x_old[2] + dyaw_dt * dt;
//             // x_k[3] = x_old[3] + dv_dt   * dt;
//             // // End of inlined logic
            
//             // // STREAM WRITE: Pack and send data
//             // TrajectoryState packet;
//             // packet.k = k;
//             // packet.t = t;
//             // for (int i = 0; i < dim_x; i++) packet.x[i] = x_k[i];
//             // for (int i = 0; i < dim_u; i++) packet.v[i] = v_k[i];
//             // trajectory_out.write(packet);
//             // ✅ Pack the calculated state into the bundle instead of writing to the stream.
//             bundle.states[t].k = k;
//             bundle.states[t].t = t;
//             for (int i = 0; i < dim_x; i++) bundle.states[t].x[i] = x_k[i];
//             for (int i = 0; i < dim_u; i++) bundle.states[t].v[i] = v_k[i];
//         }
//         // ✅ Write the single, complete bundle to the stream once all T steps are done.
//         trajectory_out.write(bundle);
//     }
// }


void simulate_trajectories_stream(
    pos_nn x0_in[dim_x], 
    control_t u[MAX_T][dim_u], 
    sample_gauss steer_samples[MAX_K*MAX_T], 
    sample_gauss accel_samples[MAX_K*MAX_T], 
    dtime_t dt,
    float p_expl,    
    hls::stream<TrajectoryPair>& trajectory_out
) {
#pragma HLS inline off
    
    
    TRAJECTORY_SIM_K: for (int k = 0; k < MAX_K; k++) {
        // #pragma HLS PIPELINE II=1
        // #pragma HLS UNROLL factor=16
        #pragma HLS PIPELINE II = 8

        pos_nn x0[dim_x];
        #pragma HLS ARRAY_PARTITION variable=x0 complete dim=1
        for(int i=0; i<dim_x; i++) {
            #pragma HLS UNROLL
            x0[i] = x0_in[i];
        }

        const float exploration_threshold = (1.0f) - p_expl;
        const float k_threshold = exploration_threshold * (MAX_K);
        pos_nn v_out_f[dim_u];
    
        pos_nn x_k[dim_x];
        #pragma HLS ARRAY_PARTITION variable=x_k complete dim=1
        
        for (int i = 0; i < dim_x; i++) { 
            #pragma HLS UNROLL
            x_k[i] = x0[i]; 
        }
        
        // ✅ Buffer compact states (no k, t)
        CompactTrajectoryState states_buffer[MAX_T];
        #pragma HLS ARRAY_PARTITION variable=states_buffer complete dim=1

        for (int t = 0; t < MAX_T; t++) {
            // #pragma HLS UNROLL factor=2
            // #pragma HLS PIPELINE
            control_t v_k[dim_u];
            if ((k) < k_threshold) {
                v_k[0] = u[t][0] + steer_samples[k*MAX_T+t];
                v_k[1] = u[t][1] + accel_samples[k*MAX_T+t];
            } else {
                v_k[0] = steer_samples[k*MAX_T+t];
                v_k[1] = accel_samples[k*MAX_T+t];
            }

            pos_nn g_result[dim_u];
            #pragma HLS ARRAY_PARTITION variable=g_result complete dim=1
            
            v_out_f[0] = v_k[0];
            v_out_f[1] = v_k[1];
            limit_fixed_safe(v_out_f, g_result);
            state_update_fixed(x_k, g_result, x_k, dt);
            
            // ✅ Store compact state (NO k, t fields)
            for (int i = 0; i < dim_x; i++) {
                states_buffer[t].x[i] = x_k[i];
            }
            for (int i = 0; i < dim_u; i++) {
                states_buffer[t].v[i] = v_k[i];
            }
        }
        
        // ✅ Write pairs (16 writes of 384 bits each, not 1 write of 8,192 bits)
        for (int pair_idx = 0; pair_idx < MAX_T/2; pair_idx++) {
            #pragma HLS UNROLL
            TrajectoryPair pair;
            pair.s[0] = states_buffer[pair_idx * 2];
            pair.s[1] = states_buffer[pair_idx * 2 + 1];
            trajectory_out.write(pair);  // 384 bits ✓
        }
    }
}

/*************************************************************************************/
/* STAGE 4: Calculate Costs (Streaming Version)                                      */
/*************************************************************************************/
// void calculate_all_costs_stream(
//     hls::stream<TrajectoryBundle>& trajectory_in, 
//     pos_nn nominal_ref_points[MAX_T][dim_x], 
//     control_t u[MAX_T][dim_u], 
//     dsp_optimal_t sigma_inv[dim_x], 
//     hls::stream<CostOutput>& cost_out
// ) {
//     // ✅ FIX: Restructure loops to eliminate the slow multiplexer.
//     // Process one full trajectory (all T steps) at a time.
//     // #pragma HLS PIPELINE II=1
//     #pragma HLS inline off
//     COST_K_LOOP: for (int k = 0; k < MAX_K; k++) {
//         #pragma HLS PIPELINE II=1
//         // Use a single register for accumulation, which is very fast.
//         // It gets reset for each new trajectory 'k'.
//         // ✅ Read the single, complete bundle from the stream.
//         TrajectoryBundle bundle = trajectory_in.read();
//         cost_w current_k_cost = 0;

//         COST_T_LOOP: for (int t = 0; t < MAX_T; t++) {
//             // #pragma HLS PIPELINE II=1
//             #pragma HLS UNROLL factor=4
            
//             // Read the next state from the stream.
//             // The stream guarantees data arrives in order (k=0,t=0), (k=0,t=1)...
//             TrajectoryState traj = bundle.states[t];
            
//             // Sanity check that our loop matches the stream's k
//             // This is for logic validation; can be removed for final synthesis.
//             // assert(traj.k == k); 

//             bool is_terminal = (t == MAX_T - 1);
//             cost_w stage_cost = calculate_stage_cost_fixed(traj.x, nominal_ref_points[t], is_terminal);
            
//             dsp_optimal_t u_sigma_inv[dim_u];
//             u_sigma_inv[0] = sigma_inv[0] * u[t][0] + sigma_inv[1] * u[t][1];
//             u_sigma_inv[1] = sigma_inv[2] * u[t][0] + sigma_inv[3] * u[t][1];
//             dsp_optimal_t dot_prod = u_sigma_inv[0] * traj.v[0] + u_sigma_inv[1] * traj.v[1];
//             cost_w control_cost = dsp_optimal_t(param_gamma) * dot_prod;
            
//             // Accumulate into the fast local register.
//             current_k_cost += stage_cost + control_cost;
//         }

//         // After all T steps for trajectory 'k' are done, write the final cost.
//         CostOutput packet;
//         packet.k = k;
//         packet.total_cost = current_k_cost;
//         cost_out.write(packet);
//     }
// }
void calculate_all_costs_stream(
    hls::stream<TrajectoryPair>& trajectory_in,
    pos_nn nominal_ref_points[MAX_T][dim_x], 
    control_t u[MAX_T][dim_u], 
    dsp_optimal_t sigma_inv[dim_x],
    float p_gamma, 
    hls::stream<CostOutput>& cost_out
) {
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable=nominal_ref_points complete dim=1    
    COST_K_LOOP: for (int k = 0; k < MAX_K; k++) {
        // #pragma HLS PIPELINE off
        // #pragma HLS UNROLL factor = 16
        #pragma HLS PIPELINE II = 8
        
        cost_w current_k_cost = 0;

        // ✅ Read pairs (not bundles)
        for (int pair_idx = 0; pair_idx < MAX_T/2; pair_idx++) {
            // #pragma HLS UNROLL
            // #pragma HLS PIPELINE
            
            TrajectoryPair pair = trajectory_in.read();  // 384 bits ✓
            
            // ✅ Process both compact states in the pair
            for (int local_t = 0; local_t < 2; local_t++) {
                // ✅ CRITICAL: Compute correct time index
                int t = pair_idx * 2 + local_t;  // 0,1,2,...,31
                const CompactTrajectoryState& traj = pair.s[local_t];
                
                bool is_terminal = (t == MAX_T - 1);
                cost_w stage_cost = calculate_stage_cost_fixed(
                    (pos_nn*)traj.x, nominal_ref_points[t], is_terminal
                );
                
                dsp_optimal_t u_sigma_inv[dim_u];
                u_sigma_inv[0] = sigma_inv[0] * u[t][0] + sigma_inv[1] * u[t][1];
                u_sigma_inv[1] = sigma_inv[2] * u[t][0] + sigma_inv[3] * u[t][1];
                dsp_optimal_t dot_prod = u_sigma_inv[0] * traj.v[0] + u_sigma_inv[1] * traj.v[1];
                cost_w control_cost = dsp_optimal_t(p_gamma) * dot_prod;
                
                current_k_cost += stage_cost + control_cost;
            }
        }

        CostOutput packet;
        packet.k = k;
        packet.total_cost = current_k_cost;
        cost_out.write(packet);
    }
}


// /*************************************************************************************/
// /* STAGE 5: Update Control Sequence (Streaming Version)                              */
// /*************************************************************************************/
// void update_control_sequence_stream(
//     hls::stream<CostOutput>& cost_in, 
//     sample_gauss steer_samples[MAX_K*MAX_T], 
//     sample_gauss accel_samples[MAX_K*MAX_T], 
//     control_t u[MAX_T][dim_u], 
//     control_t u_out[dim_u]
// ) {
//     cost_w S[MAX_K];
//     sample_w W[MAX_K];
//     #pragma HLS ARRAY_PARTITION variable=W cyclic factor=16 dim=1
    
//     // Read all costs from the stream first
//     READ_COSTS_LOOP: for (int i = 0; i < MAX_K; i++) {
//         #pragma HLS PIPELINE II=1
//         CostOutput packet = cost_in.read();
//         S[packet.k] = packet.total_cost; // Use k to write to correct index
//     }
    
//     // The rest of the function is identical to your original code
//     compute_weight_fixed(S, W);
    
//     control_t w_epsilon[MAX_T][dim_u];
//     #pragma HLS ARRAY_PARTITION variable=w_epsilon complete dim=2
//     for (int i = 0; i < MAX_T; i++) {
//         #pragma HLS UNROLL 
//         w_epsilon[i][0] = control_t(0.0f); 
//         w_epsilon[i][1] = control_t(0.0f); 
//     }

//     W_EPSILON_K:for (int k = 0; k < MAX_K; k++) {
//         #pragma HLS PIPELINE II=1
//         sample_w weight_temp = W[k];
//         W_EPSILON_T:for (int t = 0; t < MAX_T; t++) {
//             #pragma HLS UNROLL 
//             w_epsilon[t][0] += weight_temp * steer_samples[k*MAX_T+t];
//             w_epsilon[t][1] += weight_temp * accel_samples[k*MAX_T+t];
//         }
//     }
    
//     sample_f w_epsilon_flat[MAX_T*dim_u], w_epsilon_filtered[MAX_T*dim_u];
//     for (int t = 0; t < MAX_T; t++) { 
//         #pragma HLS UNROLL
//         for (int i = 0; i < dim_u; i++) {
//             w_epsilon_flat[t * dim_u + i] = w_epsilon[t][i]; 
//         } 
//     }

//     average_fixed(w_epsilon_flat, w_epsilon_filtered, 10);

//     for (int t = 0; t < MAX_T; t++) {
//         #pragma HLS PIPELINE II=1
//         for (int i = 0; i < dim_u; i++) { 
//             #pragma HLS UNROLL
//             u[t][i] += control_t(w_epsilon_filtered[t * dim_u + i]); 
//         }
//     }

//     for (int t = 0; t < MAX_T - 1; t++) { 
//         #pragma HLS UNROLL
//         for (int i = 0; i < dim_u; i++) { 
//             u_prev[t][i] = u[t + 1][i]; 
//         } 
//     }
//     for (int i = 0; i < dim_u; i++) { 
//         #pragma HLS UNROLL
//         u_prev[MAX_T - 1][i] = u[MAX_T - 1][i]; 
//     }
    
//     u_out[0] = u[0][0];
//     u_out[1] = u[0][1];
// }

/*************************************************************************************/
/* STAGE 5: Update Control Sequence (OVERLAPPED VERSION)                             */
/*************************************************************************************/

// ✅ SUB-STAGE 5A: Read costs and compute weights incrementally
void compute_weights_stream(
    hls::stream<CostOutput>& cost_in,
    float p_lambda,
    hls::stream<sample_w>& weight_out
) {
    // #pragma HLS PIPELINE II=1
    #pragma HLS inline off
    // cost_w S[MAX_K];
    // sample_w W[MAX_K];
    // #pragma HLS ARRAY_PARTITION variable=S cyclic factor=16 dim=1
    // #pragma HLS ARRAY_PARTITION variable=W cyclic factor=16 dim=1
    
    // ✅ Read all costs from the stream (this is fast, II=1)
    READ_COSTS_LOOP: for (int i = 0; i < MAX_K; i++) {
        #pragma HLS PIPELINE II=1
        CostOutput packet = cost_in.read();
        S[packet.k] = packet.total_cost;
    }
    
    // ✅ Compute weights (existing function, unchanged)
    compute_weight_fixed(S, W, p_lambda);
    
    // ✅ Stream out weights one at a time
    WRITE_WEIGHTS_LOOP: for (int k = 0; k < MAX_K; k++) {
        #pragma HLS PIPELINE II=1
        weight_out.write(W[k]);
    }
}

// ✅ SUB-STAGE 5B: Accumulate weighted noise as weights arrive
void accumulate_weighted_noise_stream(
    hls::stream<sample_w>& weight_in,
    sample_gauss steer_samples[MAX_K*MAX_T],
    sample_gauss accel_samples[MAX_K*MAX_T],
    control_t u_read[MAX_T][dim_u],
    control_t u_write[MAX_T][dim_u],
    control_t u_out[dim_u]
) {
    // #pragma HLS PIPELINE II=1
    #pragma HLS inline off
    // ✅ Initialize accumulator
    // control_t w_epsilon[MAX_T][dim_u];
    // #pragma HLS ARRAY_PARTITION variable=w_epsilon complete dim=2
    for (int i = 0; i < MAX_T; i++) {
        #pragma HLS UNROLL 
        w_epsilon[i][0] = control_t(0.0f); 
        w_epsilon[i][1] = control_t(0.0f); 
    }

    // ✅ Process weights as they arrive from Stage 5A
    W_EPSILON_K: for (int k = 0; k < MAX_K; k++) {
        // #pragma HLS PIPELINE
        
        // Read one weight from the stream (blocks until available)
        sample_w weight_temp = weight_in.read();
        
        // Accumulate the weighted noise for all time steps
        W_EPSILON_T: for (int t = 0; t < MAX_T; t++) {
            #pragma HLS PIPELINE II=1
            // #pragma HLS DEPENDENCE variable=w_epsilon inter false

            w_epsilon[t][0] += weight_temp * steer_samples[k*MAX_T+t];
            w_epsilon[t][1] += weight_temp * accel_samples[k*MAX_T+t];
        }
    }
    
    // ✅ The rest is unchanged: filter and update control
    // sample_f w_epsilon_flat[MAX_T*dim_u], w_epsilon_filtered[MAX_T*dim_u];
    for (int t = 0; t < MAX_T; t++) { 
        #pragma HLS UNROLL
        for (int i = 0; i < dim_u; i++) {
            w_epsilon_flat[t * dim_u + i] = w_epsilon[t][i]; 
        } 
    }

    average_fixed(w_epsilon_flat, w_epsilon_filtered, 10);

    // for (int t = 0; t < MAX_T; t++) {
    //     #pragma HLS PIPELINE II=1
    //     for (int i = 0; i < dim_u; i++) { 
    //         #pragma HLS UNROLL
    //         u[t][i] += control_t(w_epsilon_filtered[t * dim_u + i]); 
    //     }
    // }

    // for (int t = 0; t < MAX_T - 1; t++) { 
    //     #pragma HLS UNROLL
    //     for (int i = 0; i < dim_u; i++) { 
    //         u_prev[t][i] = u[t + 1][i]; 
    //     } 
    // }
    // for (int i = 0; i < dim_u; i++) { 
    //     #pragma HLS UNROLL
    //     u_prev[MAX_T - 1][i] = u[MAX_T - 1][i]; 
    // }
    
    // u_out[0] = u[0][0];
    // u_out[1] = u[0][1];
       // ✅ Now we read from u_read and write to u_write (two separate buffers)
    for (int t = 0; t < MAX_T; t++) {
        #pragma HLS PIPELINE II=1
        for (int i = 0; i < dim_u; i++) { 
            u_write[t][i] = u_read[t][i] + control_t(w_epsilon_filtered[t * dim_u + i]);
            //               ^^^^^^ READ from        ^^^^^^ WRITE to (different buffer)
        }
    }
    
    // // ✅ For next iteration, update u_prev from u_write
    // for (int t = 0; t < MAX_T - 1; t++) { 
    //     #pragma HLS UNROLL
    //     for (int i = 0; i < dim_u; i++) { 
    //         u_prev[t][i] = u_write[t + 1][i];   // ← Now reads from output
    //     } 
    // }
    // for (int i = 0; i < dim_u; i++) { 
    //     #pragma HLS UNROLL
    //     u_prev[MAX_T - 1][i] = u_write[MAX_T - 1][i];
    // }
    
    u_out[0] = u_write[0][0];   // ← Output from write buffer
    u_out[1] = u_write[0][1];
}