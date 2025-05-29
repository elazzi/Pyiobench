# iostat Parser and Graph Generator (`parse_iostat.py`)

## Script Overview

`parse_iostat.py` is a Python script designed to parse the output of the `iostat` command (specifically, output similar to that from `iostat -xtc <interval>`) and generate time-series graphs for each storage device found in the log. These graphs visualize various performance metrics over time, helping in the analysis of disk I/O behavior. Additionally, it generates an interactive HTML report to easily view these graphs.

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

Run the script from your command line. You can optionally provide the path to your iostat log file and an output directory for the graphs.

### Command Syntax:
```bash
python parse_iostat.py [input_file] [options]
```

### Arguments:

*   **`[input_file]`**: (Optional) Path to the iostat output file. If not specified, the script defaults to looking for a file named `iostat_output.log` in the current directory.
*   **`-o OUTPUT_DIR`, `--output_dir OUTPUT_DIR`**: (Optional) Directory where the generated graph image files (PNG) and HTML report will be saved. If not specified, files are saved in the current directory from which the script is run.

### Examples:

1.  **Process a specific iostat log file and save output to a specific directory:**
    ```bash
    python parse_iostat.py my_iostat_data.log -o ./iostat_analysis
    ```
    This command will parse `my_iostat_data.log` and save the generated graphs and HTML report into a directory named `iostat_analysis`.

2.  **Process the default iostat log file (`iostat_output.log`) and save output to the current directory:**
    ```bash
    python parse_iostat.py
    ```

3.  **Process the default iostat log file (`iostat_output.log`) and save output to a specific directory:**
    ```bash
    python parse_iostat.py -o ./iostat_report_files
    ```

## Output

The script produces the following output within the specified output directory (or current directory if none is provided):

*   **Individual PNG graph image files**: For each metric of each unique storage device detected in the iostat log, a separate PNG image file is generated.
    *   **Content of graphs**: Each graph displays the time-series evolution of a single iostat metric (e.g., `r/s`, `w/s`, `await`, `pct_util`) for a specific device. The X-axis represents the timestamp, and the Y-axis represents the metric's value.
    *   **Filenames**: These are typically in the format `{device_name}_{metric_name}_iostat.png` (e.g., `sda_r_s_iostat.png`, `sdb_pct_util_iostat.png`). Device and metric names are sanitized for filesystem compatibility (e.g., `/` becomes `_`).

*   **HTML Report (`iostat_report.html`)**: An interactive HTML file that allows you to browse the generated graphs. See the "Using the HTML Report" section below for details.

The script also prints informational messages to the console during its execution, including parsing progress, graph generation status, HTML report creation, and any warnings or errors encountered.

## Using the HTML Report

After running the script, an `iostat_report.html` file will be created in the output directory. Open this file in any modern web browser to use the interactive report:

1.  **Select Device**: Use the dropdown menu at the top of the page to choose a storage device (e.g., `sda`, `nvme0n1`).
2.  **Select Metrics**: Once a device is selected, a list of available metrics for that device will appear as checkboxes. Check the boxes for the metrics you wish to view.
3.  **View Graphs**: Click the "View Selected Graphs" button. The graphs for the chosen device and metrics will be displayed below the controls. Each graph image will be shown along with its title.

This interface allows you to dynamically explore the performance characteristics of your storage devices based on the parsed iostat data.
