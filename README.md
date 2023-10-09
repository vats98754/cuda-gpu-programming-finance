# CUDA-based Financial Modeling Project Structure

This repository contains the GPU-accelerated financial models using CUDA. The directory structure organizes the source files, headers, binaries, and data.

This project includes visualization capabilities for the financial models using OpenGL in conjunction with CUDA.

## Directory Structure

```
CUDA-Financial-Modeling/
│
├── src/
│   ├── black_scholes.cu       # Black-Scholes model implementation
│   ├── barrier_option.cu      # Barrier option pricing
│   ├── asian_option.cu        # Asian option pricing
│   └── pso_heston_calibration.cu # PSO for Heston model calibration
│
├── include/                   # Directory for header files (currently empty, but can be populated later)
│
├── bin/                       # Compiled binaries (currently empty, binaries will be placed here after compilation)
│
└── data/                      # Data files (e.g., market data for calibration)
```

## Visualization Setup

1.Ensure you have the OpenGL Utility Toolkit (GLUT) installed. This is used for the visualization.
2. Compile the CUDA code with both nvcc and a C++ compiler. For example:

## Compilation

To compile the CUDA code files, navigate to the `src` directory and use the `nvcc` compiler:

```bash
nvcc filename.cu -o ../bin/output_name
```

For example, to compile the Black-Scholes model:

```bash
nvcc black_scholes.cu -o ../bin/black_scholes
```

Replace `filename.cu` with the appropriate filename and `output_name` with the desired binary name.
