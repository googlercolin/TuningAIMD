import numpy as np

class TCP_Simulator:
    def __init__(self, alphas, betas, initial_window=[10, 1, 4]):
        self.alphas = alphas
        self.betas = betas
        self.window = np.array(initial_window, dtype=int)
        self.congestion_state = np.zeros(len(alphas), dtype=int)  # 0: Congestion Avoidance, 1: Fast Recovery
        self.ssthresh = np.inf
        self.throughputs = np.zeros(len(alphas))  # Store throughput for each user

    def construct_transition_matrix(self):
        n = len(self.alphas)
        A = np.diag(self.betas) + np.outer(self.alphas, (np.ones(n) - self.betas)) / sum(self.alphas)
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
            self.throughputs = self.window * self.alphas

            if np.all(np.abs(self.window - prev_window) < convergence_threshold):
                print("Convergence achieved.")
                break
            print(f"Iteration {i+1}: Congestion window = {self.window}")

        print("Final congestion windows (rounded to integers):", np.round(self.window).astype(int))

        # Calculate fairness and efficiency metrics
        self.calculate_metrics()

    def calculate_metrics(self):
        print("Throughputs:", self.throughputs)
        # Jain's Fairness Index
        jfi = np.sum(self.throughputs)**2 / (len(self.alphas) * np.sum(self.throughputs**2))
        print("Jain's Fairness Index:", jfi)

        # Throughput fairness
        max_throughput = np.max(self.throughputs)
        min_throughput = np.min(self.throughputs)
        throughput_fairness = min_throughput / max_throughput
        print("Throughput Fairness:", throughput_fairness)

        # Efficiency Metrics
        total_throughput = np.sum(self.throughputs)
        network_utilization = total_throughput / (len(self.alphas) * np.max(self.window))
        print("Network Throughput:", total_throughput)
        print("Network Utilization:", network_utilization)

def main():
    # Parameters for multiple users
    alphas = np.array([1, 1, 1])  # List of alpha values for each user
    betas = np.array([0.5, 0.5, 0.5])   # List of beta values for each user

    # Create TCP simulator instance
    tcp_simulator = TCP_Simulator(alphas, betas)

    # Simulate TCP
    tcp_simulator.simulate()

if __name__ == "__main__":
    main()
