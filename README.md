# Battery Benchmark Script

This Python script benchmarks battery consumption on a laptop by running CPU-intensive tasks with a configurable number of threads (1 to 8 by default) and polling the battery level during these tasks.

## Features

- Runs CPU-intensive floating-point operations.
- Configurable number of threads to stress the CPU.
- Polls and records battery percentage at regular intervals.
- Outputs benchmark results, including duration and battery consumption per test run, to both the console and a CSV file (`benchmark_results.csv`).
- Configurable via command-line arguments:
    - Number of threads for each test.
    - Battery polling interval.
    - Number of iterations for the CPU-intensive task.

## Dependencies

- Python 3.x
- `psutil`: Used for retrieving battery status. Install it via pip:
  ```bash
  pip install psutil
  ```

## How to Run

1.  **Clone the repository or download the `battery_benchmark.py` script.**
2.  **Install dependencies:**
    ```bash
    pip install psutil
    ```
3.  **Run the script from your terminal:**

    Basic usage (uses default parameters: 1, 2, 4, 8 threads; 10s poll interval; 10,000,000 iterations for CPU task):
    ```bash
    python battery_benchmark.py
    ```

    Custom usage:
    ```bash
    python battery_benchmark.py --thread_counts 1 2 --poll_interval 5 --iterations 5000000
    ```
    This example runs tests for 1 and 2 threads, polls the battery every 5 seconds, and sets the CPU task to 5,000,000 iterations.

    Command-line arguments:
    - `--thread_counts N [N ...]`: List of thread counts to benchmark (default: 1 2 4 8).
    - `--poll_interval SECONDS`: Battery polling interval in seconds (default: 10).
    - `--iterations NUMBER`: Number of iterations for the CPU intensive task (default: 10000000).

## Output

- **Console Output:** The script logs its progress, including the start and end of each test run, battery levels polled, and summary results for each configuration.
- **CSV File (`benchmark_results.csv`):** A CSV file is generated (or overwritten) in the same directory as the script. It contains the following columns for each test run:
    - `Number of Threads`
    - `Duration (s)`
    - `Initial Battery (%)`
    - `Final Battery (%)`
    - `Battery Consumed (%)`
    - `Iterations` (Number of iterations for the CPU task)

## Notes

- Battery polling accuracy and availability depend on the system and `psutil`'s ability to retrieve this information. If battery information is unavailable, relevant fields in the output will be marked as "N/A" or similar, and warnings will be logged.
- Ensure your laptop is running on battery power during the benchmark for meaningful results.
- The duration of each test run depends on the `--iterations` parameter and the system's performance.
