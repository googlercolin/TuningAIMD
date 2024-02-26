import numpy as np

class TCP_Simulator:
    def __init__(self, alphas=None, betas=None, alpha_function=None, beta_function=None, initial_window=[10, 1, 4]):
        if alphas is not None and betas is not None:
            self.alphas = alphas
            self.betas = betas
        elif alpha_function is not None and beta_function is not None:
            self.alpha_function = alpha_function
            self.beta_function = beta_function
        else:
            raise ValueError("Please provide either alphas and betas or alpha_function and beta_function.")
        
        self.window = np.array(initial_window, dtype=int)
        self.congestion_state = np.zeros(len(self.window), dtype=int)  # 0: Congestion Avoidance, 1: Fast Recovery
        self.ssthresh = np.inf
        self.throughputs = np.zeros(len(self.window))  # Store throughput for each user

    def update_alpha(self):
        if hasattr(self, 'alphas'):
            return self.alphas
        else:
            return self.alpha_function(self.window)

    def update_beta(self):
        if hasattr(self, 'betas'):
            return self.betas
        else:
            return self.beta_function(self.window)

    def construct_transition_matrix(self):
        n = len(self.window)
        A = np.diag(self.update_beta()) + np.outer(self.update_alpha(), (np.ones(n) - self.update_beta())) / sum(self.update_alpha())
        return A

    def update_congestion_window(self, A):
        if np.any(self.congestion_state == 1):  # In fast recovery phase
            self.window += 1  # Increase window by 1 for each ACK received
            lost_packet_idx = np.argmax(self.congestion_state == 1)  # Identify the packet loss
            self.window[lost_packet_idx] = self.ssthresh  # Set window to ssthresh
            self.congestion_state[self.congestion_state == 1] = 0  # Exit fast recovery
        else:  # In congestion avoidance or slow start phase
            self.window = np.dot(A, self.window)  # AIMD: Increase window by 1 for each RTT
            if np.any(self.window >= self.ssthresh):  # Enter fast recovery upon detecting packet loss
                self.ssthresh = np.max(self.window)  # Set ssthresh to the maximum window size achieved
                self.window = np.floor(self.ssthresh / 2)  # Reduce window to half of ssthresh
                self.congestion_state[self.window < self.ssthresh] = 1  # Enter fast recovery

    def simulate(self, max_iterations=50, convergence_threshold=1e-5):
        A = self.construct_transition_matrix()

        for i in range(max_iterations):
            prev_window = np.copy(self.window)
            self.update_congestion_window(A)

            # Calculate throughput for each user
            self.throughputs = self.window * self.update_alpha()

            if np.all(np.abs(self.window - prev_window) < convergence_threshold):
                print("Convergence achieved.")
                break
            print(f"Iteration {i+1}: Congestion window = {self.window}")

        print("Final congestion windows (rounded to integers):", np.round(self.window).astype(int))

        # Calculate fairness and efficiency metrics
        self.calculate_metrics()

    def calculate_metrics(self):
        print("Throughputs:", self.throughputs)
        # Efficiency Metrics
        total_throughput = np.sum(self.throughputs)
        print("Network Throughput:", total_throughput)
        
        # Jain's Fairness Index
        jfi = np.sum(self.throughputs)**2 / (len(self.window) * np.sum(self.throughputs**2))
        print("Jain's Fairness Index:", jfi)

        # Throughput fairness
        max_throughput = np.max(self.throughputs)
        min_throughput = np.min(self.throughputs)
        throughput_fairness = min_throughput / max_throughput
        print("Throughput Fairness:", throughput_fairness)


def exponential_alpha(window):
    return np.exp(0.05*window)

def exponential_beta(window):
    return np.exp(-window)

def logarithmic_alpha(window):
    return np.log(window + 1)

def logarithmic_beta(window):
    return 1 / (np.log(window + 5))

def polynomial_alpha(window):
    return 0.1 * window**3 + 0.5 * window**2 + 0.2 * window + 1

def polynomial_beta(window):
    return -0.01 * window**2 + 0.05 * window + 0.1

def sigmoid_alpha(window):
    return 1 / (1 + np.exp(-0.1 * (window - 5)))

def sigmoid_beta(window):
    return 1 / (1 + np.exp(-0.05 * (window - 7)))

def main():
    # Initial window sizes for multiple users
    initial_window = [10, 1, 4]

    # Predefined alpha and beta values
    alphas = np.array([1, 1, 1])  # List of alpha values for each user
    betas = np.array([0.5, 0.5, 0.5])   # List of beta values for each user
    tcp_simulator = TCP_Simulator(alphas=alphas, betas=betas, initial_window=initial_window)
    print("\nFixed Predefined Alphas and Betas:")
    print("Alphas:", alphas)
    print("Betas:", betas)
    tcp_simulator.simulate()

    # Create TCP simulator instances with custom alpha and beta functions
    # Exponential functions
    tcp_simulator_exp = TCP_Simulator(alpha_function=exponential_alpha, beta_function=exponential_beta, initial_window=initial_window)
    print("\nExponential Functions:")
    tcp_simulator_exp.simulate()

    # Logarithmic functions
    tcp_simulator_log = TCP_Simulator(alpha_function=logarithmic_alpha, beta_function=logarithmic_beta, initial_window=initial_window)
    print("\nLogarithmic Functions:")
    tcp_simulator_log.simulate()

    # Polynomial functions
    tcp_simulator_poly = TCP_Simulator(alpha_function=polynomial_alpha, beta_function=polynomial_beta, initial_window=initial_window)
    print("\nPolynomial Functions:")
    tcp_simulator_poly.simulate()

    # Sigmoid functions
    tcp_simulator_sig = TCP_Simulator(alpha_function=sigmoid_alpha, beta_function=sigmoid_beta, initial_window=initial_window)
    print("\nSigmoid Functions:")
    tcp_simulator_sig.simulate()

if __name__ == "__main__":
    main()
