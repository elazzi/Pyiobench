# iostat Parser and Graph Generator (`parse_iostat.py`)

## Script Overview

`parse_iostat.py` is a Python script designed to parse the output of the `iostat` command (specifically, output similar to that from `iostat -xtc <interval>`) and generate time-series graphs for each storage device found in the log. These graphs visualize various performance metrics over time, helping in the analysis of disk I/O behavior.

## Dependencies

To use this script, you need Python 3 and the following Python libraries:

*   `pandas`: For data manipulation and creating DataFrames.
*   `matplotlib`: For generating plots and graphs.

You can install these dependencies using pip:
```bash
pip install pandas matplotlib
```

## Input File Format

The script expects an input file that contains the text output of an `iostat` command run with options like `-x` (extended statistics), `-t` (timestamp), and `-c` (CPU utilization, though CPU data itself is ignored for graphing device metrics). A typical command to generate compatible output would be:

```bash
iostat -xtc <interval>
```
(e.g., `iostat -xtc 5 > my_iostat_data.log`)

Each block of device statistics in the file should be preceded by a timestamp line (e.g., `03/22/2024 10:00:00 AM`). The script is designed to be robust in handling various common timestamp formats and the typical structure of `iostat` output, including skipping `avg-cpu` sections and initial lines with system information.

## Usage

Run the script from your command line, providing the path to your iostat log file and an optional output directory for the graphs.

### Command Syntax:
```bash
python parse_iostat.py <input_file> [options]
```

### Arguments:

*   **`<input_file>`**: (Required) Path to the iostat output file.
*   **`-o OUTPUT_DIR`, `--output_dir OUTPUT_DIR`**: (Optional) Directory where the generated graph image files (PNG) will be saved. If not specified, graphs are saved in the current directory from which the script is run.

### Example:
```bash
python parse_iostat.py my_iostat_data.log -o ./iostat_graphs
```
This command will parse `my_iostat_data.log` and save the generated graphs into a directory named `iostat_graphs` (created if it doesn't exist) inside the current working directory.

## Output

The script produces the following output:

*   **PNG image files**: For each unique storage device detected in the iostat log, a separate PNG image file is generated.
*   **Content of graphs**: Each graph displays the time-series evolution of various iostat metrics (e.g., `r/s`, `w/s`, `await`, `pct_util`) for that specific device. The X-axis represents the timestamp, and the Y-axis represents the metric values.
*   **Location**: These graph files are saved in the directory specified by the `--output_dir` option (or the current directory if not specified). Filenames are typically in the format `{device_name}_iostat_metrics.png` (e.g., `sda_iostat_metrics.png`).

The script also prints informational messages to the console during its execution, including parsing progress, graph generation status, and any warnings or errors encountered.
