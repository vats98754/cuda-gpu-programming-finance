
#include <curand_kernel.h>

#define PARAMETERS 5
#define MAX_ITERATIONS 1000
#define COGNITIVE 0.5f
#define SOCIAL 0.5f
#define INERTIA 0.5f

__global__ void psoHestonCalibrationKernel(float* d_particles, float* d_velocities, float* d_best_positions, float* d_global_best_position, float* d_fit_values, float* d_market_data, int num_market_data, curandState* states)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[tid];
    curand_init(1234, tid, 0, &localState);

    float kappa = d_particles[tid * PARAMETERS + 0];
    float theta = d_particles[tid * PARAMETERS + 1];
    float xi    = d_particles[tid * PARAMETERS + 2];
    float rho   = d_particles[tid * PARAMETERS + 3];
    float v0    = d_particles[tid * PARAMETERS + 4];

    float fitness = 0.0;

    for (int i = 0; i < num_market_data; i++)
    {
        float simulated_value = kappa + theta + xi + rho + v0;
        float market_value = d_market_data[i];

        fitness += (simulated_value - market_value) * (simulated_value - market_value);
    }

    d_fit_values[tid] = fitness;

    if (fitness < d_fit_values[tid])
    {
        d_fit_values[tid] = fitness;
        for (int i = 0; i < PARAMETERS; i++)
        {
            d_best_positions[tid * PARAMETERS + i] = d_particles[tid * PARAMETERS + i];
        }
    }

    if (fitness < d_fit_values[0])
    {
        for (int i = 0; i < PARAMETERS; i++)
        {
            d_global_best_position[i] = d_particles[tid * PARAMETERS + i];
        }
    }

    for (int i = 0; i < PARAMETERS; i++)
    {
        float cognitive_component = COGNITIVE * curand_uniform(&localState) * (d_best_positions[tid * PARAMETERS + i] - d_particles[tid * PARAMETERS + i]);
        float social_component = SOCIAL * curand_uniform(&localState) * (d_global_best_position[i] - d_particles[tid * PARAMETERS + i]);

        d_velocities[tid * PARAMETERS + i] = INERTIA * d_velocities[tid * PARAMETERS + i] + cognitive_component + social_component;
        d_particles[tid * PARAMETERS + i] += d_velocities[tid * PARAMETERS + i];
    }
}
