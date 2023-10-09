
#include <math.h>

__global__ void blackScholesKernel(float* d_callResult, float* d_putResult, float* d_stockPrice, float* d_optionStrike, float* d_optionYears, float Riskfree, float Volatility, int numOptions)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    float S = d_stockPrice[idx];
    float X = d_optionStrike[idx];
    float T = d_optionYears[idx];
    float R = Riskfree;
    float V = Volatility;

    float sqrtT = sqrt(T);
    float d1 = (logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    float d2 = d1 - V * sqrtT;
    float cndd1 = 0.5f * (1.0f + erf(d1 / sqrt(2.0f)));
    float cndd2 = 0.5f * (1.0f + erf(d2 / sqrt(2.0f)));

    float expRT = expf(-R * T);
    d_callResult[idx] = (S * cndd1 - X * expRT * cndd2);
    d_putResult[idx]  = (X * expRT * (1.0f - cndd2) - S * (1.0f - cndd1));
}
