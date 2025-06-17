import math
import threading
import time
# psutil is imported conditionally in get_battery_percentage
import csv
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global event to signal threads to stop
stop_event = threading.Event()
psutil_available = False
try:
    import psutil
    psutil_available = True
    logging.info("psutil library found. Battery polling enabled.")
except ImportError:
    logging.error("psutil library not found. Battery polling is disabled. Please install psutil (pip install psutil).")


def get_battery_percentage():
  """
  Retrieves the current battery percentage.

  Returns:
    The current battery percentage as an integer, or None if not available or psutil is not installed.
  """
  if not psutil_available:
    return None

  try:
    battery = psutil.sensors_battery()
    if battery:
      logging.debug(f"Current battery: {battery.percent}%") # DEBUG for frequent polls
      return battery.percent
    else:
      logging.warning("Battery information not available on this system (psutil.sensors_battery() returned None).")
      return None
  except Exception as e:
    logging.error(f"Error getting battery status via psutil: {e}")
    return None

def cpu_intensive_task(iterations):
  """
  Performs CPU-intensive calculations for a given number of iterations.
  """
  for i in range(iterations):
    _ = math.sqrt(i)
    _ = math.pow(i, 0.5)

def run_threads(num_threads, iterations):
  """
  Creates and runs a specified number of threads that execute cpu_intensive_task.
  """
  threads = []
  for _ in range(num_threads):
    thread = threading.Thread(target=cpu_intensive_task, args=(iterations,))
    threads.append(thread)
    thread.start()

  for thread in threads:
    thread.join()

def battery_poller(polling_interval, stop_event_ref):
  """
  Polls and prints (and logs) battery status at regular intervals.
  """
  logging.info("Battery poller thread started.")
  while not stop_event_ref.is_set():
    battery_level = get_battery_percentage()
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if battery_level is not None:
      print(f"[{current_time}] Battery: {battery_level}%") # Keep console output for live monitoring
      # logging.info(f"Battery level: {battery_level}%") # Avoid duplicate logging if already in get_battery_percentage or too verbose
    else:
      print(f"[{current_time}] Battery: N/A")
      logging.info("Battery poller: No battery data available or psutil not installed.")

    stop_event_ref.wait(polling_interval)
  logging.info("Battery poller thread stopped.")

def main_benchmark(args):
  """
  Runs the main CPU benchmark with logging and command-line args.
  """
  logging.info("Starting battery benchmark.")
  logging.info(f"Running with parameters: Threads={args.thread_counts}, Poll Interval={args.poll_interval}s, Iterations/Task={args.iterations}")

  thread_counts = args.thread_counts
  polling_interval = args.poll_interval
  iterations_per_task = args.iterations
  csv_filename = "benchmark_results.csv"

  try:
    with open(csv_filename, 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(["Number of Threads", "Duration (s)", "Initial Battery (%)", "Final Battery (%)", "Battery Consumed (%)", "Iterations per Task"])
    logging.info(f"Results will be saved to {csv_filename}")
  except IOError as e:
    logging.error(f"Error initializing CSV file {csv_filename}: {e}. Results will not be saved.")
    # Consider if script should exit if CSV is critical; for now, it continues

  for num_threads in thread_counts:
    logging.info(f"--- Starting test run for {num_threads} thread(s) ---")

    stop_event.clear()

    initial_battery = get_battery_percentage()
    if initial_battery is None:
      logging.warning(f"Could not retrieve initial battery level for {num_threads} thread(s) run.")
      print("Initial battery: N/A")
    else:
      print(f"Initial battery: {initial_battery}%")

    start_time = time.time()

    poller_thread = threading.Thread(target=battery_poller, args=(polling_interval, stop_event))
    poller_thread.start()

    run_threads(num_threads, iterations_per_task)

    end_time = time.time()

    final_battery = get_battery_percentage()
    if final_battery is None:
      logging.warning(f"Could not retrieve final battery level for {num_threads} thread(s) run.")
      print("Final battery: N/A")
    else:
      print(f"Final battery: {final_battery}%")

    logging.info("Signaling battery poller to stop...")
    stop_event.set()
    poller_thread.join()

    duration = end_time - start_time
    battery_consumed = None
    if initial_battery is not None and final_battery is not None:
      battery_consumed = initial_battery - final_battery

    initial_battery_str = str(initial_battery) if initial_battery is not None else "N/A"
    final_battery_str = str(final_battery) if final_battery is not None else "N/A"
    battery_consumed_str = str(battery_consumed) if battery_consumed is not None else "N/A"

    logging.info(f"Test run for {num_threads} thread(s) completed. Duration: {duration:.2f}s, Battery consumed: {battery_consumed_str}%")
    print(f"--- Results for {num_threads} thread(s) ---") # Keep console summary
    print(f"Duration: {duration:.2f} seconds")
    if initial_battery is not None and final_battery is not None : # only print consumed if values were available
        print(f"Battery consumed: {battery_consumed_str}%")
    else:
        print(f"Battery consumption data N/A.")


    try:
      with open(csv_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([num_threads, f"{duration:.2f}", initial_battery_str, final_battery_str, battery_consumed_str, iterations_per_task])
    except IOError as e:
      logging.error(f"Error writing results to CSV file {csv_filename} for {num_threads} threads: {e}")

    if num_threads < thread_counts[-1]:
        logging.info(f"Waiting for 5 seconds before next test run...")
        time.sleep(5)

  logging.info(f"Benchmark finished. Results saved to {csv_filename (if saving was successful else 'N/A')}") # A bit more dynamic
  print("\n--- Benchmark finished ---")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="CPU Benchmark tool with battery monitoring.")
  parser.add_argument('--thread_counts', nargs='+', type=int, default=[1, 2], help="List of thread counts (e.g., 1 2 4).") # Adjusted default
  parser.add_argument('--poll_interval', type=int, default=5, help="Battery polling interval in seconds.") # Adjusted default
  parser.add_argument('--iterations', type=int, default=1_000_000, help="Iterations for CPU task.") # Adjusted default

  args = parser.parse_args()

  # psutil check is now at the top, main_benchmark will proceed with psutil_available flag
  main_benchmark(args)
