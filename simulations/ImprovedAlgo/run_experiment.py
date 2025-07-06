import random
import time
import experiments_csv
import santa_claus_solver_original
import santa_claus_solver_improved

# Function to generate a random Santa Claus problem instance
def generate_instance(InstanceClass, num_agents, num_items):
    agents = [f"Kid_{i}" for i in range(num_agents)]
    items = [f"Present_{i}" for i in range(num_items)]
    agent_item_values = {}
    for agent in agents:
        for item in items:
            # Assign random values between 0 and 100
            agent_item_values[(agent, item)] = random.randint(0, 100)
    return InstanceClass(agents, items, agent_item_values)

# Function to run a single experiment instance
def run_santa_claus_experiment(num_agents: int, num_items: int, algorithm: str):
    if algorithm == "Original":
        InstanceClass = santa_claus_solver_original.Instance
        solver_func = santa_claus_solver_original.santa_claus_solver
    elif algorithm == "Improved":
        InstanceClass = santa_claus_solver_improved.Instance
        solver_func = santa_claus_solver_improved.santa_claus_solver
    else:
        raise ValueError("Unknown algorithm")

    instance = generate_instance(InstanceClass, num_agents, num_items)

    start_time = time.time()
    min_happiness, total_happiness, _ = solver_func(instance)
    end_time = time.time()
    time_taken = end_time - start_time

    return {
        "min_happiness": min_happiness,
        "total_happiness": total_happiness,
        "time_taken": time_taken
    }

# Define input ranges for the experiment
input_ranges = {
    "num_agents": [10],
    "num_items": range(100, 2001, 100),  # Vary number of items from 100 to 2000, step 100
    "algorithm": ["Original", "Improved"]
}

# Create an Experiment object
# The first argument is the folder for results, second is the filename, third is for backups
experiment = experiments_csv.Experiment("results/", "santa_claus_performance.csv", "results/backups/")

# Run the experiment
if __name__ == "__main__":
    print("Running Santa Claus Problem Optimization Experiment...")
    experiment.run(run_santa_claus_experiment, input_ranges)
    print("Experiment finished. Results saved to santa_claus_performance.csv.")

    # Plotting the results using matplotlib directly as experiments-csv.plot_results is not available
    import pandas as pd
    import matplotlib.pyplot as plt

    try:
        df = pd.read_csv("results/santa_claus_performance.csv")

        fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        fig.suptitle("Santa Claus Problem Algorithm Comparison", fontsize=16)

        # Plot Runtime
        ax = axes[0]
        for algo in df['algorithm'].unique():
            subset = df[df['algorithm'] == algo]
            ax.plot(subset['num_items'], subset['time_taken'], label=algo)
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Runtime vs. Number of Items")
        ax.legend()
        ax.grid(True)

        # Plot Minimum Happiness
        ax = axes[1]
        for algo in df['algorithm'].unique():
            subset = df[df['algorithm'] == algo]
            ax.plot(subset['num_items'], subset['min_happiness'], label=algo)
        ax.set_xlabel("Number of Items")
        ax.set_ylabel("Minimum Happiness")
        ax.set_title("Minimum Happiness vs. Number of Items")
        ax.legend()
        ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
        plt.savefig("runtime_and_min_happiness_comparison.png")
        plt.show()

        print("Plots generated successfully.")

    except FileNotFoundError:
        print("Error: santa_claus_performance.csv not found. Run the experiment first.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
