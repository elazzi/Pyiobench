# Pyiobench - System Performance Benchmarking Suite

This suite consists of tools for benchmarking and analyzing system performance, including disk I/O and CPU performance.

## Components

### 1. CPU Benchmark (`cpu_benchmark.py`)
A tool for measuring CPU performance across different workloads:
- Multi-threaded performance testing
- Single-core performance evaluation
- CPU stress testing with configurable load levels
- Performance metrics collection and analysis

### 2. Disk Benchmark (`disk_benchmark.py`)
Evaluates storage device performance with:
- Sequential read/write tests
- Random access performance measurement
- I/O operations per second (IOPS) testing
- Block size impact analysis
- Multi-threaded I/O testing

### 3. iostat Parser and Graph Generator (`parse_iostat.py`)

## Script Overview

`parse_iostat.py` is a Python script designed to parse the output of the `iostat` command (specifically, output similar to that from `iostat -xtc <interval>`). It processes this data and generates a highly interactive HTML report. This report allows users to dynamically select and view time-series charts, rendered using D3.js, for various performance metrics from multiple storage devices simultaneously, aiding in the analysis of disk I/O behavior.

## Dependencies

To use this script, you need Python 3 and the following Python library:

*   `pandas`: For data manipulation and creating DataFrames.

You can install this dependency using pip:
```bash
pip install pandas
```
The HTML report uses the D3.js library, which is loaded via a CDN link within the HTML file. No separate installation for D3.js is needed to run the Python script.

## Input File Format

The script expects an input file that contains the text output of an `iostat` command run with options like `-x` (extended statistics), `-t` (timestamp), and `-c` (CPU utilization, though CPU data itself is ignored for charting device metrics). A typical command to generate compatible output would be:

```bash
iostat -xtc <interval>
```
(e.g., `iostat -xtc 5 > my_iostat_data.log`)

Each block of device statistics in the file should be preceded by a timestamp line (e.g., `03/22/2024 10:00:00 AM`). The script is designed to be robust in handling various common timestamp formats and the typical structure of `iostat` output, including skipping `avg-cpu` sections and initial lines with system information.

## Usage

Run the script from your command line. You can optionally provide the path to your iostat log file and an output directory.

### Command Syntax:
```bash
python parse_iostat.py [input_file] [options]
```

### Arguments:

*   **`[input_file]`**: (Optional) Path to the iostat output file. If not specified, the script defaults to looking for a file named `iostat_output.log` in the current directory.
*   **`-o OUTPUT_DIR`, `--output_dir OUTPUT_DIR`**: (Optional) Directory where the generated HTML report file will be saved. If not specified, the file is saved in the current directory from which the script is run.

### Examples:

1.  **Process a specific iostat log file and save the HTML report to a specific directory:**
    ```bash
    python parse_iostat.py my_iostat_data.log -o ./iostat_analysis
    ```
    This command will parse `my_iostat_data.log` and save the generated HTML report into a directory named `iostat_analysis`.

2.  **Process the default iostat log file (`iostat_output.log`) and save the HTML report to the current directory:**
    ```bash
    python parse_iostat.py
    ```

3.  **Process the default iostat log file (`iostat_output.log`) and save the HTML report to a specific directory:**
    ```bash
    python parse_iostat.py -o ./iostat_report_files
    ```

## Output

The script produces the following output within the specified output directory (or current directory if none is provided):

*   **HTML Report (`iostat_report.html`)**: A dynamic and interactive HTML file that renders time-series charts using D3.js. This single file contains all the data and logic to browse and visualize the iostat metrics. See the "Using the HTML Report" section below for details on its features.

The script also prints informational messages to the console during its execution, including parsing progress, chart data preparation status, HTML report creation, and any warnings or errors encountered.

## Using the HTML Report

After running the script, an `iostat_report.html` file will be created in the output directory. Open this file in any modern web browser to use the interactive report:

The report interface presents a list of all detected devices. Under each device heading, you'll find checkboxes for all available metrics for that specific device. Next to each metric checkbox, the maximum value observed for that metric in the dataset (e.g., "(Max: 123.45)" or "(Max: N/A)") is shown to help in selecting relevant charts.

*   **Interactive Chart Display**:
    *   To view a chart for a specific metric of a device, simply **check the corresponding checkbox**. The chart will appear immediately in the display area below the controls.
    *   To hide a chart, **uncheck its checkbox**. The chart will be removed from the display.
*   **Multi-Device and Multi-Metric Viewing**:
    *   You can select and view metrics from multiple different devices simultaneously. This allows for easy visual comparison of performance characteristics across devices.
    *   Similarly, you can select multiple metrics for the same device or different devices to view them all on the page at once.
*   **Chart Interactivity**: The charts are rendered using D3.js and include features like:
    *   **Zoom and Pan**: Use the mouse wheel to zoom in/out on chart areas and click-and-drag to pan across the chart. This is useful for inspecting data points more closely. (Note: Tooltip functionality for specific data points is not currently implemented in this D3 version).
*   **Layout**: Charts are arranged in a responsive grid, making it easy to view multiple plots regardless of screen size. Each chart is displayed with its title, indicating the metric and device.

This reactive interface allows you to dynamically explore the performance characteristics of your storage devices based on the parsed iostat data.

## Using the CPU Benchmark

Run the CPU benchmark with:
```bash
python cpu_benchmark.py [options]
```

### Options:
- `--threads N`: Number of threads to use (default: number of CPU cores)
- `--duration N`: Test duration in seconds (default: 60)
- `--load N`: CPU load percentage (1-100, default: 100)

### Output:
- Detailed CPU performance metrics
- Thread scaling efficiency
- Core utilization statistics
- Performance graphs (when used with parse_iostat.py)

## Using the Disk Benchmark

Run the disk benchmark with:
```bash
python disk_benchmark.py [options]
```

### Options:
- `--device PATH`: Path to device or file to test
- `--size N`: Size of test data in MB (default: 1024)
- `--block-size N`: Block size in KB (default: 4)
- `--mode {seq,random}`: Access pattern (default: seq)
- `--operation {read,write,both}`: Test operation (default: both)

### Output:
- Throughput measurements (MB/s)
- IOPS statistics
- Latency metrics
- Access pattern analysis
- Performance graphs (when used with parse_iostat.py)

## Using the iostat Parser


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
- Ensure your laptop is running on battery power during the benchmark for meaningful results.Add commentMore actions
- The duration of each test run depends on the `--iterations` parameter and the system's performance.