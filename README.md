# TuningAIMD
This project provides a Python implementation of a TCP (Transmission Control Protocol) AIMD (Additive Increase Multiplicative Decrease) simulator. AIMD is a congestion control algorithm used by TCP to avoid network congestion and ensure fair resource allocation among users. This simulator allows users to explore different alpha and beta functions to analyze their impact on TCP dynamics and network performance.

## Features
- **Customizable Functions**: Users can define custom alpha and beta functions or choose from predefined functions.
- **Multiple Users**: Simulates TCP behavior for multiple users with varying initial window sizes.
- **Efficiency Metrics**: Calculates efficiency metrics such as network throughput and utilization, fairness metrics including Jain's Fairness Index, and throughput fairness.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/googlercolin/TuningAIMD.git
cd TuningAIMD
pip install numpy
```

## Usage

### Benchmarking AIMD functions
```bash
python tcp_aimd_funcs.py
```

### Benchmarking multiple users using a single AIMD function
```bash
python tcp_aimd_users.py
```

### Standalone function usage
```python
from tcp_aimd_funcs import TCP_Simulator, exponential_alpha, exponential_beta

# Define initial window sizes for multiple users
initial_window = [10, 1, 4]

# Create TCP simulator instance with exponential functions
tcp_simulator = TCP_Simulator(alpha_function=exponential_alpha, beta_function=exponential_beta, initial_window=initial_window)

# Simulate TCP
tcp_simulator.simulate()
```

## Example
The provided examples demonstrate how to use the simulator with different alpha and beta functions, evaluate TCP behavior for varying numbers of users, and analyze network performance metrics.

## Future Research
The project acknowledges the need for future research in optimizing the selection of function hyperparameters. Techniques such as GridSearch can be employed to fine-tune hyperparameters, achieving faster convergence or higher network throughput, depending on the use case.

## License
This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.