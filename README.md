# iostat Parser and Graph Generator (`parse_iostat.py`)

## Script Overview

`parse_iostat.py` is a Python script designed to parse the output of the `iostat` command (specifically, output similar to that from `iostat -xtc <interval>`). It processes this data and generates a highly interactive HTML report. This report allows users to dynamically select and view time-series charts for various performance metrics from multiple storage devices simultaneously, aiding in the analysis of disk I/O behavior.

## Dependencies

To use this script, you need Python 3 and the following Python library:

*   `pandas`: For data manipulation and creating DataFrames.

You can install this dependency using pip:
```bash
pip install pandas
```
The HTML report uses the Chart.js library (and its zoom plugin) which are loaded via CDN links within the HTML file, so no separate installation for Chart.js is needed to run the Python script.

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

*   **HTML Report (`iostat_report.html`)**: A dynamic and interactive HTML file that renders time-series charts using Chart.js. This single file contains all the data and logic to browse and visualize the iostat metrics. See the "Using the HTML Report" section below for details on its features.

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
*   **Chart Interactivity**: The charts are rendered using Chart.js and include features like:
    *   **Tooltips**: Hover over data points to see exact values and timestamps.
    *   **Zoom and Pan**: Use the mouse wheel to zoom in/out on chart areas and click-and-drag to pan across the chart. This is useful for inspecting data points more closely.
*   **Layout**: Charts are arranged in a responsive grid, making it easy to view multiple plots regardless of screen size. Each chart is displayed with its title, indicating the metric and device.

This reactive interface allows you to dynamically explore the performance characteristics of your storage devices based on the parsed iostat data.
