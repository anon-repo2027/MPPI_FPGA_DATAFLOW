#include "globals.hpp"

// ===== VEHICLE SIMULATION FUNCTIONS (TESTBENCH ONLY) =====
float state[4];  // Local simulation state for testbench

void clip(float v[dim_u], float limit[dim_u]){
    float max_steer_float = float(max_steer_abs);
    float max_accel_float = float(max_accel_abs);
    // Clamp steering input
    if (v[0] > (max_steer_float)) {
        limit[0] = max_steer_float;
    } else if (v[0] < (-max_steer_float)) {
        limit[0] = -max_steer_float;
    } else {
        limit[0] = v[0];
    }

    // Clamp acceleration input
    if (v[1] > (max_accel_float)) {
        limit[1] = max_accel_float;
    } else if (v[1] < (-max_accel_float)) {
        limit[1] = -max_accel_float;
    } else {
        limit[1] = v[1];
    }
}

void update(float delta_t, float u[2], float default_delta_t){
    // Initialize parameters
    float l = wheel_base;
    float dt = (delta_t == 0.0) ? default_delta_t : delta_t;

    // Keep previous states
    float x = state[0];
    float y = state[1];
    float yaw = state[2];
    float v = state[3];

    // Update state variables using bicycle model
    float new_x = x + v * hls::cosf(yaw) * dt;
    float new_y = y + v * hls::sinf(yaw) * dt;
    float new_yaw = yaw + (v / l) * hls::tanf(u[0]) * dt;
    float new_v = v + u[1] * dt;

    state[0] = new_x;
    state[1] = new_y;
    state[2] = new_yaw;
    state[3] = new_v;
}

void get_state(float state_out[4]){
    for (int i = 0; i < 4; i++) {
        state_out[i] = state[i];
    }
}

void reset(const float init_state[4]) {
    for (int i = 0; i < 4; i++) {
        state[i] = init_state[i];
    }
}


void calc_control_input(float observed_x[dim_x], float ref_path_array[1201][4], float u_out[dim_u], float mppi_params[12]);

// ✅ Noise generation and saving functions
// void generategauss_fixed_point(sample_t steer_samples[MAX_K*MAX_T], sample_t accel_samples[MAX_K*MAX_T]);
// void save_noise_samples(sample_t steer_samples[MAX_K*MAX_T], sample_t accel_samples[MAX_K*MAX_T], int num_samples);




int main() {
    std::cout << "[INFO] Start simulation of pathtracking with MPPI controller" << std::endl;
    
    float mppi_params[12] = {
    // 512,
    // 16,    
    0.04f,     // dt
    0.0f,      // exploration
    100.0f,    // lambda
    0.98f,     // alpha
    50.0f, 50.0f, 1.0f, 20.0f,  // stage weights
    50.0f, 50.0f, 1.0f, 20.0f   // terminal weights
    };

    float delta_t = 0.04;
    int sim_steps = 1000;
    std::cout << "[INFO] delta_t: " << delta_t << "[s], sim_steps: " << sim_steps << "[steps], total_sim_time: " << delta_t * sim_steps << "[s]" << std::endl;

    float init_state[4] = {0.0, 0.0, 0.0, 0.0};
    reset(init_state);
    // int nearest_wp_idx = 0;
    for(int i=0;i<sim_steps;i++){
        float current_state[dim_x];
        float u_out[dim_u];
        // float u[32][2];
        get_state(current_state);

        calc_control_input(current_state, ref_path_array, u_out, mppi_params);
        // Print current state and input force
        std::cout << "Time: "<< std::fixed << std::setprecision(2) << i * delta_t << "[s], "
                  << "x="  << std::fixed << std::setprecision(5) <<current_state[0] << "[m], "
                  << "y=" << std::fixed << std::setprecision(5) << current_state[1] << "[m], "
                  << "yaw="  << std::fixed << std::setprecision(5) << current_state[2] << "[rad], "
                  << "v="  << std::fixed << std::setprecision(5) << current_state[3] << "[m/s], "
                  << "steer="  << u_out[0] << "[rad], "
                  << "accel="  << u_out[1] << "[m/s^2]" << std::endl;

        update(delta_t, u_out, 0.04);
    }

    // // ✅ NEW: Test noise generation and save to files
    // std::cout << "[INFO] Generating and saving noise samples..." << std::endl;
    
    // // Declare noise arrays
    // sample_t steer_samples[MAX_K*MAX_T];
    // sample_t accel_samples[MAX_K*MAX_T];
    
    // // Generate noise
    // generategauss_fixed_point(steer_samples, accel_samples);
    
    // // Save to files
    // save_noise_samples(steer_samples, accel_samples, MAX_K*MAX_T);
    
    // std::cout << "[INFO] Noise generation and saving completed." << std::endl;

    return 0;
}  