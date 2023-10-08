# CUDA-based Financial Modeling

Financial modeling is an essential tool in trading, risk management, and investment. With the ever-growing complexity of models and the demand for real-time analysis, it's crucial to employ high-performance computing solutions. This project leverages the power of CUDA to accelerate various financial simulations and optimizations.

![CUDA Logo](https://developer.nvidia.com/sites/default/files/pictures/2018/cuda.png)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributions](#contributions)
- [License](#license)

## Introduction

This project focuses on implementing GPU-accelerated financial models using CUDA. The power of parallel processing in GPUs allows for faster simulations and optimizations, especially beneficial for Monte Carlo methods and global optimization algorithms.

## Features

1. **Black-Scholes Model**: Compute call and put option prices using the Black-Scholes formula.
2. **Heston Model Simulation**: Stochastic volatility model for option pricing using Monte Carlo methods.
3. **Barrier Option Pricing**: Path-dependent option pricing using Monte Carlo simulations.
4. **Asian Option Pricing**: Path-dependent option whose payoff depends on the average price of the underlying asset over time.
5. **Heston Model Calibration**: Calibration of the Heston model parameters using Particle Swarm Optimization.

## Installation

### Prerequisites:

- Install the latest [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

### Steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/CUDA-Financial-Modeling.git
   ```

2. Navigate to the project directory:

   ```bash
   cd CUDA-Financial-Modeling
   ```

3. Compile the CUDA code:

   ```bash
   nvcc main.cu -o finance_modeling
   ```

## Usage

Run the compiled binary:

```bash
./finance_modeling
```

## Contributions

Contributions are always welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
