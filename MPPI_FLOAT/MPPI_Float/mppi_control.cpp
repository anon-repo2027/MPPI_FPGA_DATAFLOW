#include "globals.hpp"


struct CostOutput {
    int k;
    float total_cost;
};


// ✅ Compact state (no k, t - they're implicit in loops)
struct CompactTrajectoryState {
    float x[dim_x];      // 128 bits (4 floats)
    float v[dim_u];      // 64 bits (2 floats)
    // Total: 192 bits per state
};

// ✅ Pack 2 compact states per write
struct TrajectoryPair {
    CompactTrajectoryState s[2];
    // Total: 384 bits per pair ✓
};

// ========== FORWARD DECLARATIONS (STREAMING) ==========
void initialize_mppi_fixed(float observed_x[dim_x], float sigma_inv[dim_x], float u[MAX_T][dim_u], float x0[dim_x]);
void generategauss_fixed_point(float steer_samples[MAX_K*MAX_T], float accel_samples[MAX_K*MAX_T]);
void precalc_nominal_refs_fixed(float x0[dim_x], float u[MAX_T][dim_u], float ref_path_f32[1201][4], float nominal_ref_points[MAX_T][dim_x], float dt, int &nearest_wp_idx);

void simulate_trajectories_stream(
    float x0[dim_x], 
    float u[MAX_T][dim_u], 
    float steer_samples[MAX_K*MAX_T], 
    float accel_samples[MAX_K*MAX_T], 
    float dt,
    float p_expl, // Pass parameter
    hls::stream<TrajectoryPair>& trajectory_out
);

void calculate_all_costs_stream(
    hls::stream<TrajectoryPair>& trajectory_in, 
    float nominal_ref_points[MAX_T][dim_x], 
    float u[MAX_T][dim_u], 
    float sigma_inv[dim_x], 
    float p_gamma, // Pass parameter
    hls::stream<CostOutput>& cost_out
);

void compute_weights_stream(
    hls::stream<CostOutput>& cost_in,
    float p_lambda, // Pass parameter
    hls::stream<float>& weight_out
);

void accumulate_weighted_noise_stream(
    hls::stream<float>& weight_in,
    float steer_samples[MAX_K*MAX_T],
    float accel_samples[MAX_K*MAX_T],
    float u_read[MAX_T][dim_u],    // ← Input: what to read from
    float u_write[MAX_T][dim_u],   // ← Output: what to write to (NEW)
    float u_out[dim_u]
);

// ========== HELPER PROTOTYPES (UNCHANGED) ==========
void nearest_waypoint_fixed(float x, float y, bool update_prev_idx, float output[dim_x], float ref_path[1201][4], int &prev_idx_out);
float calculate_stage_cost_fixed(float x_t[dim_x], float ref_point[dim_x], bool is_terminal_cost);
void state_update_fixed(float x[dim_x], float g[dim_u], float result[dim_x], float dt);
void limit_fixed_safe(float v[dim_u], float result[dim_u]);
void compute_weight_fixed(float S[MAX_K], float W[MAX_K], float p_lambda);
void average_fixed(float xx[MAX_T*dim_u], float xx_avg[MAX_T*dim_u], int window_size);
void inverse2x2_fixed(float matrix[dim_x], float result[dim_x]);
float determinant2x2_fixed(float matrix[4]);
void matVecMult_fixed(float matrix[4], float vec[2], float result[2]);
float dotProduct_fixed(float vec1[2], float vec2[2]);


// Static arrays for data persistence and sharing
static float steer_samples[MAX_K*MAX_T];
static float accel_samples[MAX_K*MAX_T];
static float nominal_ref_points[MAX_T][dim_x];
static int nearest_wp_idx = 0;

// Read buffer: used by Stages 3-4
static float u_read[MAX_T][dim_u];


// Write buffer: output from Stage 5B
static float u_write[MAX_T][dim_u];


// ✅ NEW: Static buffers for intermediate results (avoid stack allocation)
static float w_epsilon[MAX_T][dim_u];
static float w_epsilon_flat[MAX_T*dim_u];
static float w_epsilon_filtered[MAX_T*dim_u];
static float S[MAX_K];
static float W[MAX_K];


// ✅ NEW: Wrapper function to handle parameter updates canonically in dataflow
void update_mppi_parameters(float p_expl, float p_lambda, float p_alpha, float s_w[dim_x], float t_w[dim_x]) {
    #pragma HLS inline
    param_exploration = p_expl;
    param_lambda = p_lambda;
    param_alpha = p_alpha;
    param_gamma = p_lambda * (1.0f - p_alpha);
    for (int i = 0; i < dim_x; i++) {
        #pragma HLS UNROLL
        stage_floateight[i] = s_w[i];
        terminal_floateight[i] = t_w[i];
    }
}

// ✅ NEW: Wrapper function to update u_prev buffer
void update_u_prev_buffer(float u_write[MAX_T][dim_u]) {
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
    
    float dt = float(all_params[0]);
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
    float x0[dim_x];
    float u[MAX_T][dim_u];
    float sigma_inv[dim_x];
    float u_internal[dim_u];
    

    update_mppi_parameters(exploration_local, lambda_local, alpha_local, stage_weights_local, terminal_weights_local);
    // Generate all noise BEFORE the dataflow region.

    generategauss_fixed_point(steer_samples, accel_samples);
 
    // STAGE 2: Calculate the constant inputs for the pipeline
    initialize_mppi_fixed(observed_x, sigma_inv, u_read, x0);

    
    #pragma HLS DATAFLOW

    hls::stream<TrajectoryPair> trajectory_stream("trajectory_stream");
    hls::stream<CostOutput> cost_stream("cost_stream");
    hls::stream<float> weight_stream("weight_stream");  // ✅ NEW stream
    
    #pragma HLS STREAM variable=trajectory_stream depth=16384
    #pragma HLS STREAM variable=cost_stream depth=2048
    #pragma HLS STREAM variable=weight_stream depth=2048    // ✅ NEW stream depth

    // STAGE 3: Pre-calculate References. Reads from u_current.
    precalc_nominal_refs_fixed(x0, u_read, ref_path_array, nominal_ref_points, dt, nearest_wp_idx);
    
    // STAGE 4: Simulate Trajectories. Reads from u_current.
    simulate_trajectories_stream(x0, u_read, steer_samples, accel_samples, dt, exploration_local, trajectory_stream);
    
    // STAGE 5: Calculate Costs. Reads from u_current.
    calculate_all_costs_stream(trajectory_stream, nominal_ref_points, u_read, sigma_inv, (lambda_local * (1.0f - alpha_local)), cost_stream);

    // STAGE 6A: Compute Weights (starts as soon as costs arrive)
    compute_weights_stream(cost_stream, lambda_local, weight_stream);
    
    // STAGE 6B: Accumulate Weighted Noise (starts as soon as weights arrive)
    accumulate_weighted_noise_stream(weight_stream, steer_samples, accel_samples, u_read, u_write, u_internal);

    // STAGE 7: Final control output limiting
    limit_fixed_safe(u_internal, u_out);

    // STAGE 8: Update the control sequence for the next iteration
    update_u_prev_buffer(u_write);
}

/*************************************************************************************/
/* Stage 1: Initialization (Updated to include ref_path conversion)                  */
/*************************************************************************************/
void initialize_mppi_fixed(float observed_x[dim_x], float sigma_inv[dim_x], float u[MAX_T][dim_u], float x0[dim_x]) {
    // #pragma HLS PIPELINE II=1
    #pragma HLS inline
    // Fixed-point sigma matrix
    float sigma[dim_x] = {
        float(0.075f), float(0.0f), 
        float(0.0f), float(2.0f)
    };
    inverse2x2_fixed(sigma, sigma_inv);

    // Initialize control sequence from previous iteration
    for (int i = 0; i < MAX_T; i++) {
        #pragma HLS UNROLL
        for (int j = 0; j < dim_u; j++) {
            u[i][j] = u_prev[i][j];
        }
    }


    for (int i = 0; i < dim_x; i++) {
        #pragma HLS UNROLL
        x0[i] = observed_x[i];
    }
    
}

/*************************************************************************************/
/* Stage 2: Pre-calculate Nominal Reference Path (Simplified)                        */
/*************************************************************************************/

void precalc_nominal_refs_fixed(
    float x0_in[dim_x], 
    float u[MAX_T][dim_u], 
    float ref_path_array[1201][4], 
    float nominal_ref_points[MAX_T][dim_x], 
    float dt, 
    int &start_wp_idx
) {
    // #pragma HLS ARRAY_PARTITION variable=ref_path_array complete dim=2
    #pragma HLS inline off
    #pragma HLS BIND_STORAGE variable=ref_path_array type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=ref_path_array cyclic factor=4 dim=2
    
    float u_fp[dim_u];
    
    // ✅ TWO buffers instead of one
    float x_current[dim_x];  // ← Read from this
    float x_next[dim_x];     // ← Write to this
    #pragma HLS ARRAY_PARTITION variable=x_current complete dim=1
    #pragma HLS ARRAY_PARTITION variable=x_next complete dim=1
    
    for(int i=0; i<dim_x; i++) {
        #pragma HLS UNROLL
        x_current[i] = x0_in[i];
    }
    
    int local_prev_idx = start_wp_idx;
    float ref_output[dim_x];
    float g_result[dim_u];
    
    // Single loop with initialization handled on first iteration
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

void simulate_trajectories_stream(
    float x0_in[dim_x], 
    float u[MAX_T][dim_u], 
    float steer_samples[MAX_K*MAX_T], 
    float accel_samples[MAX_K*MAX_T], 
    float dt,
    float p_expl,    
    hls::stream<TrajectoryPair>& trajectory_out
) {
#pragma HLS inline off
    
    
    TRAJECTORY_SIM_K: for (int k = 0; k < MAX_K; k++) {
        // #pragma HLS PIPELINE II=1
        // #pragma HLS UNROLL factor=16
        #pragma HLS PIPELINE II = 8

        float x0[dim_x];
        #pragma HLS ARRAY_PARTITION variable=x0 complete dim=1
        for(int i=0; i<dim_x; i++) {
            #pragma HLS UNROLL
            x0[i] = x0_in[i];
        }

        const float exploration_threshold = (1.0f) - p_expl;
        const float k_threshold = exploration_threshold * (MAX_K);
        float v_out_f[dim_u];
    
        float x_k[dim_x];
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
            float v_k[dim_u];
            if ((k) < k_threshold) {
                v_k[0] = u[t][0] + steer_samples[k*MAX_T+t];
                v_k[1] = u[t][1] + accel_samples[k*MAX_T+t];
            } else {
                v_k[0] = steer_samples[k*MAX_T+t];
                v_k[1] = accel_samples[k*MAX_T+t];
            }

            float g_result[dim_u];
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
void calculate_all_costs_stream(
    hls::stream<TrajectoryPair>& trajectory_in,
    float nominal_ref_points[MAX_T][dim_x], 
    float u[MAX_T][dim_u], 
    float sigma_inv[dim_x],
    float p_gamma, 
    hls::stream<CostOutput>& cost_out
) {
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable=nominal_ref_points complete dim=1    
    COST_K_LOOP: for (int k = 0; k < MAX_K; k++) {
        // #pragma HLS PIPELINE off
        // #pragma HLS UNROLL factor = 16
        #pragma HLS PIPELINE II = 8
        
        float current_k_cost = 0;

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
                float stage_cost = calculate_stage_cost_fixed(
                    (float*)traj.x, nominal_ref_points[t], is_terminal
                );
                
                float u_sigma_inv[dim_u];
                u_sigma_inv[0] = sigma_inv[0] * u[t][0] + sigma_inv[1] * u[t][1];
                u_sigma_inv[1] = sigma_inv[2] * u[t][0] + sigma_inv[3] * u[t][1];
                float dot_prod = u_sigma_inv[0] * traj.v[0] + u_sigma_inv[1] * traj.v[1];
                float control_cost = float(p_gamma) * dot_prod;
                
                current_k_cost += stage_cost + control_cost;
            }
        }

        CostOutput packet;
        packet.k = k;
        packet.total_cost = current_k_cost;
        cost_out.write(packet);
    }
}



/*************************************************************************************/
/* STAGE 5: Update Control Sequence                                                  */
/*************************************************************************************/

// ✅ SUB-STAGE 5A: Read costs and compute weights incrementally
void compute_weights_stream(
    hls::stream<CostOutput>& cost_in,
    float p_lambda,
    hls::stream<float>& weight_out
) {
    // #pragma HLS PIPELINE II=1
    #pragma HLS inline off
    // float S[MAX_K];
    // float W[MAX_K];
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
    hls::stream<float>& weight_in,
    float steer_samples[MAX_K*MAX_T],
    float accel_samples[MAX_K*MAX_T],
    float u_read[MAX_T][dim_u],
    float u_write[MAX_T][dim_u],
    float u_out[dim_u]
) {
    // #pragma HLS PIPELINE II=1
    #pragma HLS inline off
    // ✅ Initialize accumulator
    // float w_epsilon[MAX_T][dim_u];
    // #pragma HLS ARRAY_PARTITION variable=w_epsilon complete dim=2
    for (int i = 0; i < MAX_T; i++) {
        #pragma HLS UNROLL 
        w_epsilon[i][0] = float(0.0f); 
        w_epsilon[i][1] = float(0.0f); 
    }

    // ✅ Process weights as they arrive from Stage 5A
    W_EPSILON_K: for (int k = 0; k < MAX_K; k++) {
        // #pragma HLS PIPELINE
        
        // Read one weight from the stream (blocks until available)
        float weight_temp = weight_in.read();
        
        // Accumulate the weighted noise for all time steps
        W_EPSILON_T: for (int t = 0; t < MAX_T; t++) {
            #pragma HLS PIPELINE II=1
            // #pragma HLS DEPENDENCE variable=w_epsilon inter false

            w_epsilon[t][0] += weight_temp * steer_samples[k*MAX_T+t];
            w_epsilon[t][1] += weight_temp * accel_samples[k*MAX_T+t];
        }
    }
    
    // float w_epsilon_flat[MAX_T*dim_u], w_epsilon_filtered[MAX_T*dim_u];
    for (int t = 0; t < MAX_T; t++) { 
        #pragma HLS UNROLL
        for (int i = 0; i < dim_u; i++) {
            w_epsilon_flat[t * dim_u + i] = w_epsilon[t][i]; 
        } 
    }

    average_fixed(w_epsilon_flat, w_epsilon_filtered, 10);


    // We read from u_read and write to u_write (two separate buffers)
    for (int t = 0; t < MAX_T; t++) {
        #pragma HLS PIPELINE II=1
        for (int i = 0; i < dim_u; i++) { 
            u_write[t][i] = u_read[t][i] + float(w_epsilon_filtered[t * dim_u + i]);   
        }
    }
    
    // Output the first control vector from u_write buffer
    u_out[0] = u_write[0][0];   // ← Output from write buffer
    u_out[1] = u_write[0][1];
}