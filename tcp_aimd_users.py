import numpy as np

class TCP_Simulator:
    def __init__(self, alpha_function=None, beta_function=None, initial_window=[10, 1, 4]):
        self.alpha_function = alpha_function
        self.beta_function = beta_function
        
        self.window = np.array(initial_window, dtype=int)
        self.congestion_state = np.zeros(len(self.window), dtype=int)  # 0: Congestion Avoidance, 1: Fast Recovery
        self.ssthresh = np.inf
        self.throughputs = np.zeros(len(self.window))  # Store throughput for each user

    def update_alpha(self):
        return self.alpha_function(self.window)

    def update_beta(self):
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

def main():
    # Initial window sizes for multiple users (example)
    initial_window = [10, 1, 4]

    # Scale initial window sizes for 5 and 10 users
    initial_window_6_users = [10, 1, 4, 10, 1, 4]
    initial_window_9_users = [10, 1, 4, 10, 1, 4, 10, 1, 4]

    # Exponential functions for alpha and beta
    alpha_function = exponential_alpha
    beta_function = exponential_beta

    print("\nEvaluation with 3 users:")
    tcp_simulator = TCP_Simulator(alpha_function=alpha_function, beta_function=beta_function, initial_window=initial_window)
    tcp_simulator.simulate()

    print("\nEvaluation with 6 users:")
    tcp_simulator_6_users = TCP_Simulator(alpha_function=alpha_function, beta_function=beta_function, initial_window=initial_window_6_users)
    tcp_simulator_6_users.simulate()

    print("\nEvaluation with 9 users:")
    tcp_simulator_9_users = TCP_Simulator(alpha_function=alpha_function, beta_function=beta_function, initial_window=initial_window_9_users)
    tcp_simulator_9_users.simulate()

if __name__ == "__main__":
    main()
