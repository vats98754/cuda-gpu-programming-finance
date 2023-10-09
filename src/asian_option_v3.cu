
#include <curand_kernel.h>

__global__ void asianOptionKernel(float* d_s, float S0, float K, float T, float r, float sigma, int N, curandState* states)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[tid];
    curand_init(1234, tid, 0, &localState);

    float dt = T/N;
    float drift = exp((r - 0.5 * sigma * sigma) * dt);
    float random;
    float current_price = S0;

    float sum_prices = 0.0;

    for(int i = 0; i < N; i++)
    {
        random = curand_normal(&localState);
        current_price *= drift * exp(sigma * sqrt(dt) * random);
        sum_prices += current_price;
    }

    float average_price = sum_prices / N;

    d_s[tid] = max(average_price - K, 0.0f);
}
