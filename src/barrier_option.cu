
#include <curand_kernel.h>

__global__ void barrierOptionKernel(float* d_s, float S0, float K, float T, float r, float sigma, float B, bool isKnockOut, int N, curandState* states)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, tid, 0, &states[tid]);

    float dt = T/N;
    float drift = exp((r - 0.5 * sigma * sigma) * dt);
    float random;
    float current_price = S0;

    bool barrierCrossed = false;

    for(int i = 0; i < N && !barrierCrossed; i++)
    {
        random = curand_normal(&states[tid]);
        current_price *= drift * exp(sigma * sqrt(dt) * random);

        if(current_price >= B)
        {
            barrierCrossed = true;
        }
    }

    if (isKnockOut)
    {
        d_s[tid] = barrierCrossed ? 0 : max(current_price - K, 0.0f);
    }
    else  // Knock-in
    {
        d_s[tid] = barrierCrossed ? max(current_price - K, 0.0f) : 0;
    }
}
