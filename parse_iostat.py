#!/usr/bin/env python3
"""
Parses iostat output files to extract disk performance metrics and generates
time-series graphs for each device.

The script processes iostat logs where data for each timestamp is presented in blocks.
It extracts metrics like r/s, w/s, %util, etc., for each storage device.
Graphs are saved as PNG files in a specified output directory.

Basic Command-Line Usage:
    python parse_iostat.py [input_file_path] [-o <output_directory>]

If `input_file_path` is omitted, it defaults to `iostat_output.log` in the current directory.

Example:
    python parse_iostat.py ./my_iostat_log.txt -o ./iostat_graphs
    python parse_iostat.py -o ./graphs  # Processes default 'iostat_output.log'
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import pandas as pd
import os # For os.path.abspath

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
            else: #This was not a valid header line
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
                print(f"Warning: Could not convert value '{metric_value_str}' to float for metric '{metric_key}' on device '{device_name}'. Using 0.0 instead. Line: '{line}'")
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
    current_block_lines = []    # Accumulates lines belonging to the current_timestamp_str

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

def generate_graphs(df: pd.DataFrame, output_dir: str = '.'):
    """
    Generates and saves time-series graphs for each device found in the iostat DataFrame.

    For each unique device, this function creates a plot showing all its metrics
    over time. Graphs are saved as PNG files in the specified output directory.
    The filename for each graph is derived from the device name.

    Args:
        df: A pandas DataFrame containing parsed iostat data. Expected columns
            include 'timestamp', 'Device', and various metric columns.
        output_dir: The directory where graph image files (PNG) will be saved.
            Defaults to the current directory.
    """
    print("Info: Generating graphs...")
    if df.empty:
        print("Warning: No data available to generate graphs (DataFrame is empty).")
        return

    # os.makedirs is called before this function in main, or by generate_graphs if called standalone.
    # However, good to ensure it if this function could be called independently.
    # For now, assume output_dir exists as main creates it via generate_graphs.
    graph_metadata_list = [] # To store metadata of generated graphs

    unique_devices = df['Device'].unique()

    for device_name in unique_devices:
        device_df = df[df['Device'] == device_name].sort_values(by='timestamp')
        
        if device_df.empty:
            print(f"Info: No data for device {device_name} to plot (after filtering).")
            continue

        metric_columns = [col for col in device_df.columns if col not in ['timestamp', 'Device']]

        if not metric_columns:
            print(f"Warning: No metric columns found for device {device_name} to plot.")
            continue

        safe_device_name = _sanitize_filename_part(device_name)

        for metric_name in metric_columns:
            if device_df[metric_name].isnull().all() or (device_df[metric_name] == 0).all():
                print(f"Info: Skipping graph for device {device_name}, metric {metric_name} as all values are null or zero.")
                continue

            fig, ax = plt.subplots(figsize=(10, 5)) # Adjusted size for single metric
            
            ax.plot(device_df['timestamp'], device_df[metric_name], label=metric_name)
            
            ax.set_xlabel("Timestamp")
            ax.set_ylabel(metric_name) # Y-axis label is the metric name
            title = f"Metric: {metric_name} for Device {device_name}"
            ax.set_title(title)
            # No legend needed for a single metric plot unless we add more info
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            safe_metric_name = _sanitize_filename_part(metric_name)
            filename = f"{safe_device_name}_{safe_metric_name}_iostat.png"
            full_path = os.path.join(output_dir, filename)

            try:
                plt.savefig(full_path)
                print(f"Info: Graph saved for device {device_name}, metric {metric_name} to {full_path}")
                
                graph_metadata = {
                    'device': device_name,
                    'metric': metric_name,
                    'path': full_path, # Storing relative path as per spec
                    'title': title
                }
                graph_metadata_list.append(graph_metadata)
                
            except Exception as e:
                print(f"Error: Could not save graph for device {device_name}, metric {metric_name} to {full_path}: {e}")
            finally:
                plt.close(fig) # Close the figure to free memory
                
    return graph_metadata_list

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

def generate_html_report(graph_metadata_list: list[dict], output_dir: str):
    """
    Generates an HTML report to display the generated iostat graphs.

    The HTML file includes dropdowns to select devices and checkboxes for metrics,
    allowing users to dynamically view the graph images.

    Args:
        graph_metadata_list: A list of dictionaries, where each dictionary
                             contains metadata about a generated graph (device,
                             metric, path, title).
        output_dir: The directory where the HTML report file will be saved.
    """
    if not graph_metadata_list:
        print("Info: No graph metadata provided. HTML report will not be generated.")
        return

    processed_data = {}
    for item in graph_metadata_list:
        device = item['device']
        metric = item['metric']
        # Adjust path for HTML: make it relative to the HTML file location
        # The HTML file is in output_dir, images are also in output_dir.
        # So, path should be just the filename.
        html_path = os.path.basename(item['path'])
        
        if device not in processed_data:
            processed_data[device] = {}
        processed_data[device][metric] = {'path': html_path, 'title': item['title']}

    # Serialize processed_data to JSON for embedding in JavaScript
    # Using a simple replace for quotes in JSON to avoid issues with Python's string formatting
    # A more robust way would be to ensure no newlines/etc., or use a template engine.
    json_data = json.dumps(processed_data).replace("'", "\\'") 


    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>iostat Metrics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        h1 {{ text-align: center; color: #2c3e50; }}
        .controls {{ margin-bottom: 20px; padding: 15px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; align-items: center; gap: 15px; }}
        .controls label {{ font-weight: bold; margin-right: 5px; }}
        .controls select, .controls button {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }}
        .controls button {{ background-color: #3498db; color: white; cursor: pointer; transition: background-color 0.3s; }}
        .controls button:hover {{ background-color: #2980b9; }}
        #metricCheckboxes {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; max-height: 100px; overflow-y: auto; padding: 5px; border: 1px solid #eee; border-radius: 4px; background: #f9f9f9; min-width: 200px;}}
        #metricCheckboxes label {{ margin-right: 10px; font-weight: normal; display: flex; align-items: center;}}
        #metricCheckboxes input[type="checkbox"] {{ margin-right: 5px; }}
        #graphDisplayArea {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); /* Responsive grid */
            gap: 20px;
            padding-top: 20px;
        }}
        .graph-container {{
            background-color: #fff;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center; /* Center title and image */
        }}
        .graph-container img {{ max-width: 100%; height: auto; border-radius: 4px; margin-top:10px; }}
        .graph-container p {{ font-size: 0.9em; color: #555; margin-bottom: 10px; font-weight: bold;}}
    </style>
</head>
<body>
    <h1>iostat Metrics Report</h1>

    <div class="controls">
        <label for="deviceSelector">Select Device:</label>
        <select id="deviceSelector"></select>
        
        <label>Select Metrics:</label>
        <div id="metricCheckboxes">
            <!-- Checkboxes will be populated here -->
        </div>
        
        <button id="viewGraphsButton">View Selected Graphs</button>
    </div>

    <div id="graphDisplayArea">
        <!-- Graphs will be displayed here -->
    </div>

    <script>
        const graphData = JSON.parse('{json_data}');

        const deviceSelector = document.getElementById('deviceSelector');
        const metricCheckboxes = document.getElementById('metricCheckboxes');
        const viewGraphsButton = document.getElementById('viewGraphsButton');
        const graphDisplayArea = document.getElementById('graphDisplayArea');

        // Populate device selector on page load
        function populateDeviceSelector() {{
            const devices = Object.keys(graphData);
            if (devices.length === 0) {{
                deviceSelector.innerHTML = '<option value="">No devices found</option>';
                return;
            }}
            devices.forEach(device => {{
                const option = document.createElement('option');
                option.value = device;
                option.textContent = device;
                deviceSelector.appendChild(option);
            }});
            // Trigger metric population for the first device (if any)
            if (devices.length > 0) {{
                populateMetricCheckboxes(devices[0]);
            }}
        }}

        // Populate metric checkboxes based on selected device
        function populateMetricCheckboxes(selectedDevice) {{
            metricCheckboxes.innerHTML = ''; // Clear previous checkboxes
            if (!selectedDevice || !graphData[selectedDevice]) {{
                metricCheckboxes.innerHTML = '<span>Select a device to see metrics.</span>';
                return;
            }}

            const metrics = Object.keys(graphData[selectedDevice]);
            if (metrics.length === 0) {{
                 metricCheckboxes.innerHTML = '<span>No metrics found for this device.</span>';
                 return;
            }}
            metrics.forEach(metric => {{
                const checkboxId = `metric-${selectedDevice}-${metric}`; // Unique ID
                const checkboxLabel = document.createElement('label');
                checkboxLabel.setAttribute('for', checkboxId);
                
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = checkboxId;
                checkbox.value = metric;
                checkbox.name = 'metric';
                // checkbox.checked = true; // Optional: check all by default

                checkboxLabel.appendChild(checkbox);
                checkboxLabel.appendChild(document.createTextNode(metric));
                metricCheckboxes.appendChild(checkboxLabel);
            }});
        }}

        // Event listener for device selector change
        deviceSelector.addEventListener('change', (event) => {{
            populateMetricCheckboxes(event.target.value);
        }});

        // Event listener for "View Selected Graphs" button
        viewGraphsButton.addEventListener('click', () => {{
            graphDisplayArea.innerHTML = ''; // Clear previous graphs
            const selectedDevice = deviceSelector.value;
            
            if (!selectedDevice || !graphData[selectedDevice]) {{
                alert('Please select a device.');
                return;
            }}

            const selectedMetrics = [];
            metricCheckboxes.querySelectorAll('input[type="checkbox"]:checked').forEach(checkbox => {{
                selectedMetrics.push(checkbox.value);
            }});

            if (selectedMetrics.length === 0) {{
                alert('Please select at least one metric.');
                return;
            }}

            selectedMetrics.forEach(metric => {{
                if (graphData[selectedDevice][metric]) {{
                    const item = graphData[selectedDevice][metric];
                    
                    const graphContainer = document.createElement('div');
                    graphContainer.className = 'graph-container';
                    
                    const titleElement = document.createElement('p');
                    titleElement.textContent = item.title;
                    graphContainer.appendChild(titleElement);
                    
                    const img = document.createElement('img');
                    img.src = item.path; // Path is already relative to output_dir
                    img.alt = item.title;
                    graphContainer.appendChild(img);
                    
                    graphDisplayArea.appendChild(graphContainer);
                }}
            }});
        }});

        // Initial population
        populateDeviceSelector();
    </script>
</body>
</html>
    """

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
    1. Parses command-line arguments (input file path and optional output directory).
    2. Calls `parse_iostat_file` to process the iostat log.
    3. If parsing is successful and data is returned, calls `generate_graphs`
       to create and save plots.
    4. Handles potential errors like file not found or other exceptions during execution.
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
        help="Directory to save the generated graph image files. Defaults to the current directory."
    )
    
    args = parser.parse_args()

    try:
        # parse_iostat_file will print its own "Info: Parsing..." message
        iostat_df = parse_iostat_file(args.input_file)

        if iostat_df is not None and not iostat_df.empty:
            print(f"Info: Successfully parsed data. Found {len(iostat_df)} entries.")
            # generate_graphs will print its own "Info: Generating graphs..."
            graph_metadata = generate_graphs(iostat_df, args.output_dir)
            
            if graph_metadata: # If graphs were generated, create an HTML report
                generate_html_report(graph_metadata, args.output_dir)
                print(f"Info: Script finished successfully. Graphs and HTML report saved to {os.path.abspath(args.output_dir)}")
            else:
                print("Info: No graphs were generated, so no HTML report will be created.")
                print(f"Info: Script finished. Check warnings if data was expected. Output directory: {os.path.abspath(args.output_dir)}")

        elif iostat_df is not None and iostat_df.empty:
            # This case implies parsing happened but yielded no data
            print("Warning: No data parsed from the file. File might be empty, not in iostat format, or all data was malformed.")
            print("Info: No graphs will be generated.")
        else: 
            # This case implies a failure within parse_iostat_file that led to an empty DataFrame
            # but wasn't an outright exception (e.g. read error but not FileNotFoundError).
            print("Error: Failed to parse iostat data. No DataFrame was returned or an unexpected error occurred during parsing.")
            print("Info: No graphs will be generated.")

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
