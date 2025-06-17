"""
FP64 Monte Carlo CPU Benchmark Script

This script benchmarks the CPU's performance on fp64 (double-precision)
floating-point operations, designed to be representative of workloads found
in Monte Carlo simulations.

The benchmark involves generating arrays of random numbers and performing a
sequence of mathematical operations on them (logarithm, exponentiation,
square root, power, and basic arithmetic). This is intended to simulate
computationally intensive tasks that are common in scientific computing and
financial modeling.

Features:
- Utilizes NumPy for efficient array-based fp64 computations.
- Simulates a series of mathematical operations (log, exp, sqrt, power, arithmetic)
  on arrays of random numbers.
- Supports multi-threading via concurrent.futures.ThreadPoolExecutor to assess
  performance scaling on multi-core processors.
- Configurable parameters via command-line arguments:
    - Number of simulation paths (iterations).
    - Size of data arrays per path.
    - Number of threads.
- Outputs performance metrics: total time, paths per second, estimated
  Millions of Operations Per Second (MOPS), and GigaFLOPs (GFLOPs).
- Includes basic logging for execution tracing and error handling.

Dependencies:
- Python 3.x
- NumPy (install via: pip install numpy)

Basic Usage:
  python fp64_montecarlo_benchmark.py --num_paths 1000 --data_size 100000 --threads 4

This command runs the benchmark with 1000 simulation paths, each processing an
array of 100,000 random numbers, using 4 threads.

Interpreting FLOPs:
The script estimates MOPS/GFLOPs based on a predefined factor representing the
approximate number of floating-point operations performed per element in each
simulation path's core computation. This factor is defined by the constant
`FLOPS_PER_ELEMENT_IN_STEP` in the script (currently set to 10).
The operations counted are:
  - v1 = np.log(random_numbers + 1e-9): 1 add, 1 log = 2 FLOPs
  - v2 = np.exp(v1 / 2.0):             1 div, 1 exp = 2 FLOPs
  - v3 = np.sqrt(v2 * 0.5):            1 mul, 1 sqrt = 2 FLOPs
  - v4 = np.power(v3, 1.5):            1 pow = 1 FLOP
  - v5 = v1 + v2 - v3 * v4:            1 mul, 1 add, 1 sub = 3 FLOPs
Total = 10 FLOPs per element.
Generation of random numbers and the final sum reduction are not included in this
FLOP count for the purpose of this specific CPU-bound arithmetic benchmark.
"""

import time
import concurrent.futures
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Attempt to import NumPy and exit if not found
try:
    import numpy as np
except ImportError:
    logging.error("NumPy library not found. This script requires NumPy. Please install it (e.g., pip install numpy).")
    exit(1)

# Constant defining the estimated number of floating-point operations per element
# processed in the core computation of `monte_carlo_simulation_step`.
# See module docstring for breakdown.
FLOPS_PER_ELEMENT_IN_STEP = 10

def monte_carlo_simulation_step(data_size: int) -> float:
  """
  Performs one step of a Monte Carlo-like simulation using FP64 operations.

  This function generates an array of `data_size` random numbers and applies
  a sequence of NumPy-based vectorized mathematical operations to them.
  The operations are designed to be CPU-bound and use double-precision (fp64).

  Args:
    data_size: The number of random numbers to generate and process in the array.
               This directly impacts the computational load of this step.

  Returns:
    The sum of all elements in the final resulting array after all operations.
    This is a single float value.

  Raises:
    MemoryError: If `data_size` is too large and NumPy cannot allocate memory
                 for the arrays.
    Exception: Other exceptions from NumPy operations if inputs are invalid
               (though current design minimizes this).
  """
  try:
    # Generate fp64 random numbers (default for np.random.rand)
    random_numbers = np.random.rand(data_size)

    # Sequence of FP64 operations
    v1 = np.log(random_numbers + 1e-9)  # Add epsilon to prevent log(0)
    v2 = np.exp(v1 / 2.0)
    v3 = np.sqrt(v2 * 0.5)
    v4 = np.power(v3, 1.5)
    v5 = v1 + v2 - v3 * v4

    result_sum = np.sum(v5) # Sum reduction
    return float(result_sum)
  except MemoryError:
    logging.error(f"MemoryError in monte_carlo_simulation_step with data_size {data_size}. Try reducing data_size.", exc_info=True)
    raise
  except Exception as e:
    logging.error(f"Exception in monte_carlo_simulation_step: {e}", exc_info=True)
    raise

def worker_task(num_worker_paths: int, data_size_per_path: int) -> float:
  """
  Worker function executed by each thread in multi-threaded mode.

  This function runs `monte_carlo_simulation_step` multiple times
  (`num_worker_paths`) and aggregates their results.

  Args:
    num_worker_paths: The number of simulation paths (i.e., calls to
                      `monte_carlo_simulation_step`) this worker should execute.
    data_size_per_path: The data size (number of elements) to be used for
                        each call to `monte_carlo_simulation_step`.

  Returns:
    The sum of results from all simulation paths executed by this worker.
    This is a single float value.
  """
  local_sum = 0.0
  for i in range(num_worker_paths):
    try:
      local_sum += monte_carlo_simulation_step(data_size_per_path)
    except Exception:
      # Error is already logged in monte_carlo_simulation_step.
      # Worker task will continue with other paths if possible, effectively skipping this one's result.
      # Depending on requirements, one might want to re-raise to stop the entire benchmark.
      logging.warning(f"Skipping path {i+1}/{num_worker_paths} in worker due to error.")
  return local_sum

def run_benchmark(num_paths: int, data_size_per_path: int, num_threads: int) -> tuple[float, float]:
  """
  Runs the main Monte Carlo benchmark.

  This function orchestrates the execution of `monte_carlo_simulation_step`
  for a total of `num_paths`, distributing the work across `num_threads`
  if `num_threads > 1`. It measures the total execution time.

  Args:
    num_paths: The total number of simulation paths to execute.
    data_size_per_path: The data size for each individual simulation path.
    num_threads: The number of worker threads to use. If 1, runs in the
                 current thread without ThreadPoolExecutor.

  Returns:
    A tuple containing:
      - duration (float): The total time taken for the benchmark in seconds.
      - total_sum (float): The aggregated sum of results from all executed paths.
  """
  total_sum = 0.0
  start_time = time.perf_counter()

  if num_threads <= 1:
    logging.info(f"Running benchmark in single-threaded mode ({num_paths} paths).")
    total_sum = worker_task(num_paths, data_size_per_path)
  else:
    logging.info(f"Running benchmark in multi-threaded mode with {num_threads} threads ({num_paths} paths).")

    # Calculate number of paths for each thread, distributing any remainder
    paths_per_thread_base = num_paths // num_threads
    remainder_paths = num_paths % num_threads

    tasks_for_each_thread = []
    for i in range(num_threads):
        paths_for_this_thread = paths_per_thread_base
        if i < remainder_paths:
            paths_for_this_thread += 1
        if paths_for_this_thread > 0: # Add task only if there are paths to process
            tasks_for_each_thread.append(paths_for_this_thread)

    futures = []
    # concurrent.futures is part of stdlib, less critical for explicit ImportError check than numpy
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
      for i, paths_count in enumerate(tasks_for_each_thread):
        logging.debug(f"Submitting task of {paths_count} paths to thread {i+1}.")
        futures.append(executor.submit(worker_task, paths_count, data_size_per_path))

      # Retrieve results from completed futures
      for future in concurrent.futures.as_completed(futures):
        try:
          total_sum += future.result()
        except Exception as e:
          # This catches errors from worker_task or monte_carlo_simulation_step if they propagate
          logging.error(f"A worker thread task resulted in an exception: {e}", exc_info=True)
          # Depending on strategy, might re-raise or sum partial results. Current: sum successful.

  end_time = time.perf_counter()
  duration = end_time - start_time

  logging.info(f"Benchmark run phase completed. Aggregated sum: {total_sum:.4e}")
  return duration, total_sum

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description="FP64 Monte Carlo CPU Benchmark Script using NumPy.",
      formatter_class=argparse.RawTextHelpFormatter # To allow for newlines in help
  )
  parser.add_argument(
      '--num_paths',
      type=int,
      default=100,
      help="Number of simulation paths (iterations) to run.\n(default: 100)"
  )
  parser.add_argument(
      '--data_size',
      type=int,
      default=100000,
      help="Size of the data array (number of random numbers) per path.\n(default: 100000)"
  )
  parser.add_argument(
      '--threads',
      type=int,
      default=1,
      help="Number of threads to use. Set to 1 for single-threaded execution.\n(default: 1)"
  )
  args = parser.parse_args()

  logging.info("="*50)
  logging.info("FP64 Monte Carlo CPU Benchmark")
  logging.info("="*50)
  logging.info("Configuration:")
  logging.info(f"  Total paths: {args.num_paths}")
  logging.info(f"  Data size per path: {args.data_size}")
  logging.info(f"  Threads to be used: {args.threads}")
  logging.info(f"  Estimated FLOPs per element per step: {FLOPS_PER_ELEMENT_IN_STEP}")
  logging.info("-"*50)

  try:
    duration, total_sum_from_run = run_benchmark(args.num_paths, args.data_size, args.threads)

    logging.info("-"*50)
    logging.info("Benchmark Execution Summary:")
    logging.info(f"  Total execution time: {duration:.4f} seconds")
    logging.info(f"  Aggregated sum (for validation): {total_sum_from_run:.4e}")

    if duration > 0:
        paths_per_second = args.num_paths / duration
        # Calculate total floating point operations based on paths, data size, and FLOPs factor
        total_fp_operations = args.num_paths * args.data_size * FLOPS_PER_ELEMENT_IN_STEP

        mops = total_fp_operations / duration / 1_000_000  # Millions of Operations Per Second
        gflops = mops / 1000 # GigaFLOPs

        logging.info("  --- Performance Metrics ---")
        logging.info(f"  Paths per second: {paths_per_second:.2f} paths/s")
        logging.info(f"  Estimated MOPS (Millions of Ops/sec): {mops:.4f} MOPS")
        logging.info(f"  Estimated GFLOPs (Giga Ops/sec): {gflops:.4f} GFLOPs")
    else:
        logging.warning("Duration was zero or negative. Performance metrics cannot be calculated.")
    logging.info("-"*50)

    # Optional: Compare with single-thread if multiple threads were used
    if args.threads > 1:
        logging.info("Running single-thread comparison for reference...")
        try:
            duration_single, sum_single = run_benchmark(args.num_paths, args.data_size, 1) # Run with 1 thread
            logging.info(f"  Single-thread duration: {duration_single:.4f} seconds")
            # Log sum for consistency check, e.g., if it should match multi-threaded sum
            logging.info(f"  Single-thread sum (for validation): {sum_single:.4e}")
            if duration_single > 0 and duration > 0: # Ensure both durations are valid for comparison
                speedup = duration_single / duration
                efficiency = (speedup / args.threads) * 100
                logging.info(f"  Speedup with {args.threads} threads vs 1 thread: {speedup:.2f}x")
                logging.info(f"  Parallel Efficiency: {efficiency:.2f}%")
            else:
                logging.warning("Could not calculate speedup due to zero duration in one of the runs.")
        except Exception as e_single:
            logging.error(f"An error occurred during the single-thread comparison benchmark: {e_single}", exc_info=True)
        logging.info("-"*50)

  except MemoryError:
        logging.critical(f"Benchmark failed due to MemoryError. This might happen if --data_size is too large.", exc_info=True)
  except Exception as e:
    # Catch any other unexpected errors during the main benchmark execution
    logging.critical(f"A critical error occurred during the benchmark execution: {e}", exc_info=True)

  logging.info("Benchmark script finished.")
