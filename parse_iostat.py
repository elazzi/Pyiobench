#!/usr/bin/env python3
"""
Parses iostat output files to extract disk performance metrics and generates
time-series graphs for each device.

The script processes iostat logs where data for each timestamp is presented in blocks.
It extracts metrics like r/s, w/s, %util, etc., for each storage device.
Graphs are saved as PNG files in a specified output directory.

Basic Command-Line Usage:
    python parse_iostat.py [input_file_path] [-o <output_directory>] [-xd <device1> <device2> ...]

If `input_file_path` is omitted, it defaults to `iostat_output.log` in the current directory.

Example:
    python parse_iostat.py ./my_iostat_log.txt -o ./iostat_graphs
    python parse_iostat.py -o ./graphs  # Processes default 'iostat_output.log'
    python parse_iostat.py iostat.log -o reports -xd sda sdb loop0 # Excludes sda, sdb, loop0
"""

import re
import pandas as pd
import argparse
from datetime import datetime
import os # For os.path.abspath
import json # For HTML report generation

def _parse_timestamp(timestamp_str: str) -> datetime | None:
    """
    Attempts to parse a timestamp string into a datetime object.

    It tries a list of common timestamp formats found in iostat outputs.
    If parsing fails for all known formats, a warning is printed, and None is returned.

    Args:
        timestamp_str: The timestamp string to parse (e.g., "03/22/2024 10:00:00 AM").

    Returns:
        A datetime object if parsing is successful, otherwise None.
    """
    # A list of datetime format strings to try for parsing the timestamp.
    # This list covers various common formats from different iostat versions and locales.
    formats_to_try = [
        "%m/%d/%Y %I:%M:%S %p",  # e.g., "03/22/2024 10:00:00 AM" (Locale dependent AM/PM)
        "%m/%d/%y %I:%M:%S %p",  # e.g., "03/22/24 10:00:00 AM" (Locale dependent AM/PM)
        "%m/%d/%Y %H:%M:%S",     # e.g., 03/22/2024 10:00:00 (24-hour)
        "%m/%d/%y %H:%M:%S",     # e.g., 03/22/24 10:00:00 (24-hour)
        "%Y-%m-%d %H:%M:%S",     # ISO-like e.g. 2024-03-22 10:00:00
        "%Y-%m-%dT%H:%M:%S",    # ISO 8601 e.g. 2024-03-22T10:00:00
        "%d/%m/%Y %H:%M:%S",     # e.g., 22/03/2024 10:00:00
        "%d/%m/%y %H:%M:%S",     # e.g., 22/03/24 10:00:00
        "%d.%m.%Y %H:%M:%S",     # e.g., 22.03.2024 10:00:00
        "%d.%m.%y %H:%M:%S",     # e.g., 22.03.24 10:00:00
        "%b %d %Y %I:%M:%S %p",  # e.g., Mar 22 2024 10:00:00 AM (GNU iostat with LC_TIME=C)
        "%b %d %Y %H:%M:%S",    # e.g., Mar 22 2024 10:00:00 (GNU iostat with LC_TIME=C, 24hr)
        "%x %X",                 # Locale’s appropriate date and time representation - good as a fallback
    ]
    for fmt in formats_to_try:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    print(f"Warning: Could not parse timestamp: '{timestamp_str}'. Tried formats: {formats_to_try}")
    return None

def _parse_block(timestamp_str: str, block_content: str) -> list[dict]:
    """
    Parses a single block of iostat output associated with a specific timestamp.

    A block typically contains CPU statistics (avg-cpu) followed by device statistics.
    This function identifies the header line for device data, then parses each
    subsequent line as metrics for a specific device.

    Args:
        timestamp_str: The timestamp string for this block (used for warning messages).
        block_content: A string containing all lines of the iostat output for this block.

    Returns:
        A list of dictionaries, where each dictionary contains the metrics for one
        device from this block. Returns an empty list if the block is unparseable
        or contains no device data.
    """
    parsed_timestamp = _parse_timestamp(timestamp_str)
    if not parsed_timestamp:
        # If the timestamp string itself couldn't be parsed by _parse_timestamp,
        # log it and skip this block as we can't associate data reliably.
        print(f"Warning: Skipping block due to unparseable timestamp: '{timestamp_str}'")
        return []

    lines = block_content.strip().split('\n')
    device_data_list = []
    headers = []

    device_section_started = False
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Device:") or line.startswith("Device "): # Handle "Device:" or "Device  r/s ..."
            # Found header line
            headers = re.split(r'\s+', line) # Split by any whitespace
            # Some iostat versions might have "Device:" as the first token, others just "Device"
            if headers[0] == "Device:" or headers[0] == "Device":
                # Normalize column names, remove problematic characters if any for dict keys
                headers = [h.replace('%', 'pct_') for h in headers] # pct_util instead of %util
            else: #This was not a valid header line:
                headers = [] # Reset headers if it's not a proper device line
                continue
            device_section_started = True
            continue

        if not device_section_started:
            # Lines before the "Device:" header are typically avg-cpu stats,
            # Linux version information, or empty lines. These should be skipped.
            if "avg-cpu" in line.lower() or \
               "linux version" in line.lower().strip() or \
               line.strip().startswith("Time:"): # Some iostat versions might include "Time: HH:MM:SS" lines
                continue
            if not line.strip(): # Skip any other blank lines
                continue
            # For debugging, uncomment below to see what other lines are being skipped:
            # print(f"Debug: Skipping pre-header line: '{line}'")
            continue

        # We are now in the section where device data is expected.
        if not headers:
            # This state should ideally not be reached if "Device:" line was present and parsed.
            # If it is, it means we encountered a line that looks like data but had no preceding header.
            # Warn if the line seems to contain data (has digits).
            if line.strip() and any(char.isdigit() for char in line):
                print(f"Warning: Skipping data line due to missing headers (was 'Device:' line found and parsed for this block?): '{line}'")
            continue

        # Split the data line into values based on whitespace.
        values = re.split(r'\s+', line)
        if len(values) != len(headers):
            # If the number of values doesn't match the number of headers,
            # this line is likely malformed or not a standard device data line.
            # Print a warning if it looks like it was intended to be data.
            if len(values) > 1 and any(v.replace('.', '', 1).replace(',', '', 1).isdigit() for v in values[1:]):
                print(f"Warning: Skipping malformed data line (column count mismatch: {len(values)} fields, expected {len(headers)}). Line: '{line}'")
            continue # Skip this malformed line


        device_name = values[0]
        # Defensive check: ensure device name is not something like "avg-cpu:" or "Device:"
        # which might happen if parsing logic is imperfect or iostat output is unusual.
        if device_name in ["avg-cpu:", "Device:"]:
            continue

        # Prepare a dictionary to store metrics for the current device and timestamp.
        device_metrics = {'timestamp': parsed_timestamp, 'Device': device_name}
        valid_metric_found = False # Flag to ensure we add this dict only if it has valid metrics

        # Iterate through the headers (skipping the first 'Device' header)
        # and extract corresponding values.
        for i, header_name in enumerate(headers[1:]): # headers[0] is 'Device'
            metric_key = header_name # This is already normalized (e.g. pct_util from '%util')
            try:
                metric_value_str = values[i+1] # Corresponding value based on header index
                # Convert metric value to float. Handle comma as decimal separator if present.
                metric_value = float(metric_value_str.replace(',', '.'))
                device_metrics[metric_key] = metric_value
                valid_metric_found = True
            except ValueError:
                # If conversion to float fails, log a warning and use 0.0 as a default.
                print(f"Warning: Could not convert value  to float for metric '{metric_key}' on device '{device_name}'. Using 0.0 instead. Line: '{line}'")
                device_metrics[metric_key] = 0.0
            except IndexError:
                # If a value is missing for a header, log a warning and use 0.0.
                print(f"Warning: Missing value for metric '{metric_key}' on device '{device_name}'. Using 0.0 instead. Line: '{line}'")
                device_metrics[metric_key] = 0.0

        # Add the parsed metrics for this device to the list if valid metrics were found
        # and the device name is not "avg-cpu" (a final safeguard).
        if valid_metric_found and device_name:
            if "avg-cpu" not in device_name and device_name.strip(): # Ensure device_name is not empty
                device_data_list.append(device_metrics)
            elif "avg-cpu" in device_name:
                # This case should ideally be caught earlier, but acts as a silent safeguard.
                pass

    return device_data_list

def parse_iostat_file(file_path: str) -> pd.DataFrame:
    """
    Parses the iostat output file, extracts device statistics for each timestamp,
    and returns the data as a pandas DataFrame.

    The function reads the entire file, then splits it into blocks based on
    timestamp lines. Each block is subsequently parsed to extract device-specific
    metrics. It handles various timestamp formats and normalizes column names
    (e.g., '%' to 'pct_').

    Args:
        file_path: The path to the iostat output file.

    Returns:
        A pandas DataFrame where each row represents a single device's statistics
        at a specific timestamp. Columns include 'timestamp', 'Device', and various
        iostat metrics (e.g., 'r/s', 'w/s', 'pct_util').
        Returns an empty DataFrame if the file cannot be read, is empty, or contains
        no parsable iostat data.
    """
    all_device_data = []  # List to store all parsed data dictionaries
    print(f"Info: Parsing iostat file: {file_path}...")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        # Re-raise for main() to handle with a cleaner message to the user
        raise
    except Exception as e:
        print(f"Error: Could not read file {file_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame for other read errors

    # Regex to identify timestamp lines. This pattern aims to be flexible,
    # matching common iostat timestamp formats (date followed by time, optional AM/PM).
    # Example: "03/22/2024 10:00:00 AM", "03/22/24 10:00:00"
    timestamp_pattern = r"(\d{2}/\d{2}/\d{2,4}\s+\d{2}:\d{2}:\d{2}(\s+(AM|PM))?)"

    # Split the file content by timestamp lines.
    # The `re.split` function with a capturing group keeps the delimiters (timestamps)
    # in the resulting list, which is crucial for associating data blocks with their timestamps.
    parts = re.split(f'({timestamp_pattern})', content)

    current_timestamp_str = None  # Holds the timestamp for the current block being processed
    current_block_lines = []      # Accumulates lines belonging to the current_timestamp_str

    # Iterate through the parts. Parts will alternate between timestamps and blocks of text.
    for part in parts:
        if not part or part.isspace(): # Skip empty or whitespace-only parts resulting from split
            continue

        # Check if the current part is a recognized timestamp
        stripped_part = part.strip()
        if re.fullmatch(timestamp_pattern, stripped_part):
            # If a previous block exists (we have a timestamp and accumulated lines), process it.
            if current_timestamp_str and current_block_lines:
                block_text = "\n".join(current_block_lines)
                parsed_data = _parse_block(current_timestamp_str, block_text)
                all_device_data.extend(parsed_data)

            # Start a new block
            current_timestamp_str = stripped_part
            current_block_lines = []
        else:
            # This part is not a timestamp, so it's part of the current block's content
            current_block_lines.append(part)

    # Process the very last block in the file (if any)
    if current_timestamp_str and current_block_lines:
        block_text = "\n".join(current_block_lines)
        parsed_data = _parse_block(current_timestamp_str, block_text)
        all_device_data.extend(parsed_data)

    if not all_device_data:
        # No data could be parsed, return an empty DataFrame
        return pd.DataFrame()

    # Convert the list of dictionaries (each representing a device's stats at a timestamp)
    # into a pandas DataFrame.
    df = pd.DataFrame(all_device_data)

    # Ensure the 'timestamp' column is of datetime type for proper sorting and plotting.
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df

def prepare_chart_data(df: pd.DataFrame) -> list[dict]:
    """
    Prepares data for charting, extracting time-series data for each metric
    of each device.

    Args:
        df: A pandas DataFrame containing parsed iostat data. Expected columns
            include 'timestamp', 'Device', and various metric columns.

    Returns:
        A list of dictionaries, where each dictionary contains metadata and
        series data for a specific metric of a specific device, ready for
        JavaScript charting.
        Returns an empty list if the input DataFrame is empty.
    """
    print("Info: Preparing chart data...")
    if df.empty:
        print("Warning: No data available to prepare for charts (DataFrame is empty).")
        return []

    chart_data_list = []

    unique_devices = df['Device'].unique()

    for device_name in unique_devices:
        device_df = df[df['Device'] == device_name].sort_values(by='timestamp')

        if device_df.empty:
            # This might happen if a device was present but all its data rows were filtered out due to exclusion.
            # print(f"Info: No data for device {device_name} to plot (after filtering).") # Already handled if df is empty before this loop
            continue

        metric_columns = [col for col in device_df.columns if col not in ['timestamp', 'Device']]

        if not metric_columns:
            print(f"Warning: No metric columns found for device {device_name}.")
            continue

        # safe_device_name = _sanitize_filename_part(device_name) # Not needed for chart data prep

        for metric_name in metric_columns:
            # Skip if all values are null or zero - though JS charting might handle this,
            # it can be good to preemptively filter.
            # if device_df[metric_name].isnull().all() or (device_df[metric_name] == 0).all():
            #     # print(f"Info: Skipping data preparation for device {device_name}, metric {metric_name} as all values are null or zero.")
            #     continue # Reduced verbosity here, the main check is if chart_data_list is empty later

            title = f"Metric: {metric_name} for Device {device_name}"

            max_val = device_df[metric_name].max()
            if pd.isna(max_val):
                max_val_for_json = None
            elif isinstance(max_val, (float, int)):
                max_val_for_json = max_val
            else:
                max_val_for_json = float(max_val)

            # Prepare series data
            timestamps_iso = device_df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
            values_list = [v if pd.notna(v) else None for v in device_df[metric_name].tolist()]

            metric_chart_data = {
                'device': device_name,
                'metric': metric_name,
                'title': title,
                'max_value': max_val_for_json,
                'series_data': {
                    'timestamps': timestamps_iso,
                    'values': values_list
                }
            }
            chart_data_list.append(metric_chart_data)

    return chart_data_list

def _sanitize_filename_part(part_name: str) -> str:
    """
    Sanitizes a string part for use in a filename.
    Replaces common problematic characters like '/' and '.' with underscores.
    Allows alphanumeric characters, underscores, and hyphens.
    """
    # Replace specific known problematic characters
    name = part_name.replace('/', '_').replace('%', 'pct')
    # Remove any characters not in a whitelist (alphanumeric, underscore, hyphen, dot for extension)
    # This regex keeps only safe characters.
    name = re.sub(r'[^a-zA-Z0-9_.-]+', '_', name)
    # Ensure it doesn't start or end with problematic chars like '.' or '_' if it's not the only char
    name = name.strip('._')
    if not name: # if string becomes empty after sanitization
        return "unknown"
    return name

def generate_html_report(chart_data_list: list[dict], output_dir: str):
    """
    Generates an HTML report to display charts based on the prepared data.

    The HTML file includes controls to select devices and metrics,
    and will use a JavaScript charting library to render the graphs.

    Args:
        chart_data_list: A list of dictionaries, where each dictionary
                             contains metadata and series data for a chart.
        output_dir: The directory where the HTML report file will be saved.
    """
    if not chart_data_list:
        print("Info: No chart data provided. HTML report will not be generated.")
        return

    processed_data = {}
    for item in chart_data_list: # Iterate over the new chart_data_list
        device = item['device']
        metric = item['metric']

        if device not in processed_data:
            processed_data[device] = {}

        # New structure for processed_data, including series_data
        processed_data[device][metric] = {
            'title': item['title'],
            'max_value': item.get('max_value'), # Already JSON-friendly
            'series_data': item['series_data']  # Directly pass the series_data dict
        }

    # Serialize processed_data to JSON for embedding in JavaScript
    json_string_for_template = json.dumps(processed_data)
    # Escape backslashes and single quotes for safe embedding within JavaScript single quotes
    escaped_json_string = json_string_for_template.replace('\\', '\\\\').replace("'", "\\'")

    # Determine the directory of the currently running script
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: # __file__ is not defined if, for example, run in an interactive interpreter
        script_dir = os.getcwd()
        print("Warning: __file__ not defined, assuming template is in current working directory.")

    template_path = os.path.join(script_dir, 'report_template.html')

    try:
        with open(template_path, 'r') as f:
            html_template_content = f.read()
    except FileNotFoundError:
        print(f"Error: HTML template file not found at {template_path}")
        print("Please ensure 'report_template.html' is in the same directory as the script.")
        return
    except Exception as e:
        print(f"Error reading HTML template file {template_path}: {e}")
        return

    # Inject the JSON data into the template
    html_content = html_template_content.replace('__GRAPH_DATA_JSON__', escaped_json_string)

    report_path = os.path.join(output_dir, 'iostat_report.html')
    try:
        with open(report_path, 'w') as f:
            f.write(html_content)
        print(f"Info: HTML report saved to {report_path}")
    except Exception as e:
        print(f"Error: Could not save HTML report to {report_path}: {e}")

def main():
    """
    Main execution function for the script.
    This function orchestrates the script's workflow:

    1. Parses command-line arguments (input file path, optional output directory, and excluded devices).
    2. Calls `parse_iostat_file` to process the iostat log.
    3. Filters out excluded devices from the parsed data.
    4. If data remains, calls `prepare_chart_data` and `generate_html_report`.
    5. Handles potential errors like file not found or other exceptions during execution.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Parse iostat output and generate graphs for device metrics.",
        formatter_class=argparse.RawTextHelpFormatter # To allow for newlines in help text
    )
    parser.add_argument(
        "input_file",
        nargs='?',  # Makes it optional
        default='iostat_output.log',  # Default value
        type=str,
        help="Path to the iostat output file. Defaults to 'iostat_output.log' if not provided."
    )
    parser.add_argument(
        "-o", "--output_dir",
        default=".",
        help="Directory to save the generated HTML report file. Defaults to the current directory."
    )
    parser.add_argument(
        "-xd", "--exclude-devices",
        nargs='*', # 0 or more device names
        default=[], # Default to an empty list (no exclusions)
        type=str,
        help="List of device names to exclude from the report (e.g., -xd sda loop0 nvme0n1p1)."
    )

    args = parser.parse_args()

    try:
        # parse_iostat_file will print its own "Info: Parsing..." message
        iostat_df = parse_iostat_file(args.input_file)

        if iostat_df is not None and not iostat_df.empty:
            print(f"Info: Successfully parsed data. Found {len(iostat_df)} total entries before exclusion.")

            if args.exclude_devices:
                devices_before_exclusion = set(iostat_df['Device'].unique())
                iostat_df = iostat_df[~iostat_df['Device'].isin(args.exclude_devices)]
                devices_after_exclusion = set(iostat_df['Device'].unique())
                
                excluded_actually_found = devices_before_exclusion.intersection(set(args.exclude_devices))
                
                if excluded_actually_found:
                    print(f"Info: Attempted to exclude: {', '.join(args.exclude_devices)}.")
                    print(f"Info: Successfully excluded data for device(s): {', '.join(sorted(list(excluded_actually_found)))}.")
                else:
                    print(f"Info: No specified devices for exclusion ({', '.join(args.exclude_devices)}) were found in the parsed data.")


                if iostat_df.empty:
                    print("Warning: All data was excluded or no data remained after applying exclusions.")
                    print(f"Info: Script finished. No report generated. Output directory: {os.path.abspath(args.output_dir)}")
                    return # Exit if no data left

            if iostat_df.empty: # Check again in case exclusion made it empty
                print("Info: No data available after potential exclusions. No report will be generated.")
                print(f"Info: Script finished. Output directory: {os.path.abspath(args.output_dir)}")
                return

            print(f"Info: Proceeding with {len(iostat_df['Device'].unique())} device(s) and {len(iostat_df)} entries for report generation.")
            # Ensure iostat_df is a DataFrame, not a Series
            if isinstance(iostat_df, pd.Series):
                iostat_df = iostat_df.to_frame().T
            chart_data = prepare_chart_data(iostat_df)

            if chart_data:
                # Ensure output directory exists for the HTML report
                os.makedirs(args.output_dir, exist_ok=True)
                generate_html_report(chart_data, args.output_dir)
                print(f"Info: Script finished successfully. HTML report potentially saved to {os.path.abspath(args.output_dir)}")
            else:
                print("Info: No chart data was prepared (e.g., all metrics were zero/null for remaining devices), so no HTML report will be created.")
                print(f"Info: Script finished. Check warnings if data was expected. Output directory: {os.path.abspath(args.output_dir)}")

        elif iostat_df is not None and iostat_df.empty:
            # This case implies parsing happened but yielded no data initially
            print("Warning: No data parsed from the file. File might be empty, not in iostat format, or all data was malformed.")
            print("Info: No report will be generated.")
        else:
            # This case implies a failure within parse_iostat_file that led to an empty DataFrame
            # but wasn't an outright exception (e.g. read error but not FileNotFoundError).
            print("Error: Failed to parse iostat data. No DataFrame was returned or an unexpected error occurred during parsing.")
            print("Info: No report will be generated.")

    except FileNotFoundError:
        print(f"Error: Input file not found: '{args.input_file}'. Please check the path.")
    except Exception as e:
        print(f"Error: An unexpected error occurred in main execution: {e}")
        # For debugging, consider:
        # import traceback
        # traceback.print_exc()

# Standard Python idiom to execute main() when the script is run directly
if __name__ == "__main__":
    main()
