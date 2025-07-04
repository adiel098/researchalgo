# Santa Claus Problem - Performance Comparison Experiments

This directory contains code for benchmarking and comparing different algorithms for the Santa Claus Problem on random inputs of increasing sizes.

## Overview

The implementation compares the following algorithms:

1. **Main Algorithm**: The O(log log m / log log log m) approximation algorithm with different α parameters:
   - α=2.0 - Less aggressive threshold for large gifts
   - α=3.0 - Default parameter
   - α=4.0 - More aggressive threshold for large gifts

2. **Greedy Algorithms**:
   - **Max Value Greedy**: Assigns each present to the kid who values it most
   - **Min Value Greedy**: Focuses on maximizing the minimum value by assigning presents to kids with lowest total value

3. **Simple Baselines**:
   - **Round Robin**: Simple cyclic allocation of presents to kids
   - **Random**: Random allocation of presents to kids

## Metrics

The experiments measure and compare the following metrics:

1. **Solution Quality**:
   - **Minimum Happiness**: The minimum value received by any kid (the main objective in the Santa Claus problem)
   - **Total Happiness**: The sum of values across all kids
   - **Fairness Ratio**: The ratio of minimum to maximum happiness

2. **Performance**:
   - **Runtime**: Execution time in seconds

## Running the Experiments

To run the experiments, make sure the `csv-experiments` library is installed:

```bash
pip install csv-experiments
```

Then run:

```bash
python run_experiments.py
```

This will:
1. Generate random instances of increasing sizes
2. Run each algorithm on each instance with a timeout
3. Measure and record solution quality and runtime
4. Generate plots comparing the algorithms

## Results

Results are stored in the `results` subdirectory, including:

1. CSV files with raw experiment data
2. PNG plots showing:
   - Runtime vs Instance Size
   - Min Happiness vs Instance Size
   - Total Happiness vs Instance Size
   - Fairness Ratio vs Instance Size
   - Average Allocation Size vs Instance Size

## Interpreting the Plots

- **Runtime**: Lower is better (faster execution)
- **Min Happiness**: Higher is better (main objective of the Santa Claus problem)
- **Total Happiness**: Higher is better (secondary objective)
- **Fairness Ratio**: Higher is better (closer to 1 means more equal allocation)

## Expected Outcome

We expect the main algorithm to achieve better minimum happiness (maximin objective) than the greedy and baseline algorithms, at the cost of longer runtime. The α parameter allows trading off between solution quality and runtime.
