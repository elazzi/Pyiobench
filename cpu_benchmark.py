import time
import math
import os
import sys
try:
    import psutil
except ImportError:
    print("Warning: psutil module not found. Some features like CPU affinity and detailed core info will be disabled.", file=sys.stderr)
    psutil = None # Gracefully handle if psutil is not installed

# Imports for threading, argparse will be used later
import threading
import argparse


def set_affinity(cpu_index: int) -> bool:
    """
    Sets CPU affinity for the current process to a specific core.
    Returns True if successful, False otherwise.
    """
    if not psutil:
        print("Info: psutil not available, cannot set CPU affinity.", file=sys.stderr)
        return False
    try:
        p = psutil.Process()
        p.cpu_affinity([cpu_index])
        # print(f"Affinity set to CPU {cpu_index}") # Can be verbose, enable if needed
        return True
    except psutil.AccessDenied:
        print(f"Error: Access denied when setting affinity for CPU {cpu_index}.", file=sys.stderr)
    except psutil.NoSuchProcess:
        print("Error: Current process not found (should not happen).", file=sys.stderr)
    except ValueError as e:
        print(f"Error: Invalid CPU index {cpu_index} or system does not support affinity setting. ({e})", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while setting CPU affinity for CPU {cpu_index}: {e}", file=sys.stderr)
    return False

def reset_affinity() -> bool:
    """
    Resets CPU affinity for the current process to all available CPUs.
    Returns True if successful, False otherwise.
    """
    if not psutil:
        # print("Info: psutil not available, cannot reset CPU affinity.", file=sys.stderr) # Less critical than set_affinity failure
        return False # No action needed if psutil isn't there
    try:
        p = psutil.Process()
        num_cores_for_reset = None
        if psutil:
            num_cores_for_reset = psutil.cpu_count(logical=True)

        if num_cores_for_reset is None: # Fallback if psutil not available or psutil.cpu_count failed
            num_cores_for_reset = os.cpu_count()

        if num_cores_for_reset is None: # Further fallback if os.cpu_count also failed
            print("Warning: Could not determine CPU count for resetting affinity. Defaulting to 1 core ([0]).", file=sys.stderr)
            num_cores_for_reset = 1
        
        all_cpus = list(range(num_cores_for_reset))
        if not all_cpus: # If num_cores_for_reset was 0
            print("Warning: CPU count determined as 0 for resetting affinity. Attempting with [0].", file=sys.stderr)
            all_cpus = [0] # Ensure the list is not empty for psutil.cpu_affinity

        p.cpu_affinity(all_cpus)
        # print("Affinity reset to all available CPUs.") # Can be verbose
        return True
    except psutil.AccessDenied:
        print("Error: Access denied when resetting affinity.", file=sys.stderr)
    except psutil.NoSuchProcess:
        print("Error: Current process not found (should not happen).", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while resetting CPU affinity: {e}", file=sys.stderr)
    return False


# --- Per-Thread Affinity and Benchmark Target ---
def set_current_thread_affinity(cpu_index: int) -> bool:
    """
    Sets CPU affinity for the calling thread.
    """
    if not psutil and os.name != 'nt': # psutil needed for Linux if os.sched_setaffinity is not used directly or preferred
        print("Info: psutil not available, and not on Windows. Cannot set thread CPU affinity for non-Windows without os.sched_setaffinity or psutil.", file=sys.stderr)
        # For non-Windows, if psutil is missing, os.sched_setaffinity is the prime candidate.
        # If we decide to *only* use os.sched_setaffinity for Linux, this check changes.
        # Let's assume for now that if psutil is missing, we rely on os.sched_setaffinity for Linux.

    thread_id = threading.get_native_id() # Get a unique ID for the current thread for logging

    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32

            # Ensure DWORD_PTR is defined (it's ULONG_PTR which is c_size_t)
            if not hasattr(wintypes, 'DWORD_PTR'):
                wintypes.DWORD_PTR = ctypes.c_size_t

            # Define argtypes and restype for SetThreadAffinityMask and GetCurrentThread
            kernel32.GetCurrentThread.restype = wintypes.HANDLE
            kernel32.SetThreadAffinityMask.argtypes = [wintypes.HANDLE, wintypes.DWORD_PTR]
            kernel32.SetThreadAffinityMask.restype = wintypes.DWORD_PTR # Returns previous mask
            kernel32.GetLastError.restype = wintypes.DWORD

            thread_handle = kernel32.GetCurrentThread()
            if not thread_handle: # Should not happen for a live thread
                print(f"Thread {thread_id}: Failed to get current thread handle.", file=sys.stderr)
                return False

            affinity_mask_val = 1 << cpu_index
            dw_affinity_mask = wintypes.DWORD_PTR(affinity_mask_val)

            # Clear last error before calling API that uses it, to be safe
            kernel32.SetLastError(0)
            previous_mask = kernel32.SetThreadAffinityMask(thread_handle, dw_affinity_mask)

            # SetThreadAffinityMask returns the previous mask if successful, or 0 if it fails.
            if previous_mask != 0:
                # print(f"Thread {thread_id}: Affinity successfully set to CPU {cpu_index}")
                return True
            else:
                error_code = kernel32.GetLastError()
                error_message = ctypes.WinError(error_code).strerror
                print(f"Thread {thread_id}: Failed to set affinity to CPU {cpu_index}. "
                      f"SetThreadAffinityMask returned 0. Win32 Error Code: {error_code} ({error_message})", file=sys.stderr)
                return False
        else: # Assuming Linux or similar POSIX system
            # os.sched_setaffinity takes pid (0 for current thread) and a list/iterable of CPU indices
            os.sched_setaffinity(0, [cpu_index])
            # print(f"Thread {thread_id}: Affinity successfully set to CPU {cpu_index} using os.sched_setaffinity")
            return True
    except AttributeError as e: # os.sched_setaffinity might not be available on all POSIX-like, or ctypes issues
        print(f"Thread {thread_id}: Failed to set affinity for CPU {cpu_index}. System call or attribute error: {e}", file=sys.stderr)
        return False
    except OSError as e: # For os.sched_setaffinity errors (e.g., invalid CPU index, permissions)
        print(f"Thread {thread_id}: Failed to set affinity for CPU {cpu_index}. OSError: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Thread {thread_id}: An unexpected error occurred while setting thread affinity for CPU {cpu_index}: {e}", file=sys.stderr)
        return False

def benchmark_thread_target(cpu_index: int, benchmark_type: str, duration_sec: int, results_list: list):
    """
    Target function for benchmark threads. Sets affinity and runs a benchmark.
    """
    if not set_current_thread_affinity(cpu_index):
        print(f"Thread for CPU {cpu_index} ({benchmark_type}): Failed to set thread affinity. Test may not be accurate.", file=sys.stderr)
        # Proceed with benchmark anyway, but it won't be pinned.
        # Alternatively, one could append an error and return. For now, run it unpinned.
        # results_list.append((cpu_index, 0, benchmark_type, "Affinity Error"))
        # return

    # Short delay to allow affinity to potentially take effect if there are OS scheduling latencies.
    # This is speculative and might not be necessary or effective on all OSes.
    time.sleep(0.01)


    ops = 0
    if benchmark_type == "integer":
        ops = run_integer_benchmark(duration_sec)
    elif benchmark_type == "float":
        ops = run_float_benchmark(duration_sec)
    else:
        results_list.append((cpu_index, 0, benchmark_type, "Unknown Benchmark Type"))
        return

    results_list.append((cpu_index, ops, benchmark_type))


def get_core_info() -> dict:
    """
    Retrieves information about CPU cores using psutil.
    """
    if not psutil:
        print("Warning: psutil module not found. Core information will be limited.", file=sys.stderr)
        # Fallback to os.cpu_count() if available, otherwise unknown
        num_logical_cores = os.cpu_count() if hasattr(os, 'cpu_count') else "Unknown"
        return {
            "logical_cores": num_logical_cores,
            "physical_cores": "Unknown (psutil not available)"
        }

    num_logical_cores = psutil.cpu_count(logical=True)
    num_physical_cores = psutil.cpu_count(logical=False)

    if num_physical_cores is None:
        print("Warning: Could not determine the number of physical cores. Using logical core count as fallback for physical cores.", file=sys.stderr)
        num_physical_cores = num_logical_cores
        # Alternatively, could be set to "Unknown" or some other indicator.
        # For now, assuming logical as a fallback is a common approach if detailed SMT info isn't critical.

    return {
        "logical_cores": num_logical_cores,
        "physical_cores": num_physical_cores
    }

def run_integer_benchmark(duration_seconds: float) -> int:
    """
    Runs a synthetic CPU benchmark focused on integer operations.
    """
    operations = 0
    start_time = time.monotonic()

    # Initial values, these assignments are not counted in ops
    a_val = 1234567890 
    b_val = 9876543210

    while time.monotonic() - start_time < duration_seconds:
        # Use local variables for calculations within the loop to avoid large numbers in 'operations' affecting logic
        # The 'operations' variable itself is just a counter for benchmarked ops.
        # We use a simple incrementing value for variety in calculations.
        """ loop_iter_val = operations // 20 # Roughly, to keep it small and changing

        a = a_val + loop_iter_val # 1. Addition
        b = b_val - loop_iter_val # 2. Subtraction
        c = a * b                 # 3. Multiplication
        # Ensure d_divisor is not zero; (operations + 1) is safe as operations >= 0
        d_divisor = loop_iter_val + 1 # 4. Addition
        d = c // d_divisor        # 5. Integer Division

        # Bitwise operations
        shift_amount = (loop_iter_val % 4) + 1 # 6. Modulo, 7. Addition
        e = a << shift_amount # 8. Left Shift
        f = b >> shift_amount # 9. Right Shift
        g = e ^ f             # 10. XOR

        # Some more arbitrary operations to increase workload
        # (a + b + c + d + e + f + g) -> 6 additions (11-16)
        # (loop_iter_val % 10 + 1) -> 1 modulo, 1 addition (17-18)
        # ... // ... -> 1 integer division (19)
        h_sum = a + b + c + d + e + f + g 
        h_divisor = (loop_iter_val % 10) + 1
        h = h_sum // h_divisor
        i = h & 0xFFFFFF          # 20. AND
        j = ~i                    # 21. Bitwise NOT """
        a = a_val + operations # 1. Addition
        operations += 3 # Total operations in this loop iteration

    return operations

def run_float_benchmark(duration_seconds: float) -> int:
    """
    Runs a synthetic CPU benchmark focused on floating-point operations.
    """
    operations = 0
    start_time = time.monotonic()

    # Initial values, these assignments are not counted in ops
    x_val = 123.456789
    y_val = 987.654321

    while time.monotonic() - start_time < duration_seconds:
        # Use a scaled version of a simple increment for variety in float calculations
        """loop_iter_float = float(operations // 20) # (integer division, float conversion, float division) - these are setup, not counted in the 20 ops below

        x = x_val + loop_iter_float # 1. Addition
        y = y_val - loop_iter_float # 2. Subtraction

        z = x * y                   # 3. Multiplication
        # Ensure w_divisor is not zero
        w_divisor_val = loop_iter_float + 0.000001 # 4. Addition
        w = z / w_divisor_val       # 5. Division

        # Math functions
        # Use abs to ensure sqrt input is non-negative, though w could be negative
        abs_w = abs(w)              # 6. Absolute value
        sqrt_arg = abs_w + 0.000001 # 7. Addition
        u = math.sqrt(sqrt_arg)     # 8. Square root
        
        sin_x = math.sin(x)         # 9. Sine
        cos_y = math.cos(y)         # 10. Cosine
        v = sin_x + cos_y           # 11. Addition

        # Some more arbitrary operations
        # (x + y + z + w + u + v) -> 5 additions (12-16)
        p_numerator = x + y + z + w + u + v
        # (float(loop_iter_val_int % 100) + 0.001) -> 1 int modulo, (float conv - setup), 1 float addition (17-18)
        p_divisor_val = float((operations // 20) % 100) + 0.001 # loop_iter_val_int for modulo
        p = p_numerator / p_divisor_val # 19. Division
        q = math.atan2(p, u)        # 20. Arctangent2 """
        #loop_iter_float = float(operations // 20)
        x_valx = x_val + y_val
        operations += 2 # Total operations in this loop iteration

    return operations
def perform_individual_core_tests(core_info: dict, test_duration_sec: int, benchmark_to_run: str):
    """
    Performs integer and/or float benchmarks on each logical core individually.
    'benchmark_to_run' can be "all", "integer", or "float".
    """
    print(f"\n==== Individual Core Performance Tests (Duration: {test_duration_sec}s, Type: {benchmark_to_run}) ====")

    num_logical_cores = core_info.get('logical_cores')
    if not isinstance(num_logical_cores, int) or num_logical_cores <= 0:
        print("Error: Valid number of logical cores not found in core_info. Skipping individual core tests.", file=sys.stderr)
        if psutil is None:
             print("Info: psutil is not installed, which is likely why core detection failed or was limited.", file=sys.stderr)
        return

    original_affinity_set_successfully = True
    if psutil:
        try:
            p = psutil.Process()
            original_affinity = p.cpu_affinity()
        except Exception as e:
            print(f"Warning: Could not get original CPU affinity: {e}. Will attempt to reset to all cores later.", file=sys.stderr)
            original_affinity_set_successfully = False
            original_affinity = [] # Fallback
    else: # psutil not available
        original_affinity_set_successfully = False # Cannot manage affinity
        original_affinity = []


    for cpu_index in range(num_logical_cores):
        print(f"\nTesting CPU {cpu_index}...")
        if not set_affinity(cpu_index):
            print(f"Skipping tests for CPU {cpu_index} due to affinity setting error.")
            continue

        if benchmark_to_run == "all" or benchmark_to_run == "integer":
            print(f"  Running integer benchmark on CPU {cpu_index} for {test_duration_sec}s...")
            int_ops = run_integer_benchmark(test_duration_sec)
            int_ops_per_sec = int_ops / test_duration_sec if test_duration_sec > 0 else 0
            print(f"  Integer Ops/sec on CPU {cpu_index}: {int_ops_per_sec:,.2f}")

        if benchmark_to_run == "all" or benchmark_to_run == "float":
            print(f"  Running float benchmark on CPU {cpu_index} for {test_duration_sec}s...")
            float_ops = run_float_benchmark(test_duration_sec)
            float_ops_per_sec = float_ops / test_duration_sec if test_duration_sec > 0 else 0
            print(f"  Float Ops/sec on CPU {cpu_index}: {float_ops_per_sec:,.2f}")

    # Reset affinity to original state or all cores
    if psutil: # Only attempt if psutil was available
        if original_affinity_set_successfully and original_affinity:
            try:
                p = psutil.Process()
                p.cpu_affinity(original_affinity)
                # print("CPU affinity restored to original settings.")
            except Exception as e:
                print(f"Error restoring original CPU affinity: {e}. Attempting to reset to all cores.", file=sys.stderr)
                reset_affinity() # Try resetting to all if restoring original failed
        else:
            # If original affinity wasn't fetched, or psutil just became available.
            reset_affinity()
    print("--- Individual Core Performance Tests Finished ---")


def main():
    """
    Main function to orchestrate CPU benchmarks.
    """
    # For now, using fixed values. Will integrate argparse later.
    # args = parse_arguments()
    test_duration_per_core = 5 # Using a shorter duration for now, can be 60s as per plan

    print("Starting CPU Benchmark Suite...\n")

    core_info = get_core_info()
    print("--- System Core Information ---")
    if core_info:
        print(f"  Logical cores detected: {core_info.get('logical_cores', 'N/A')}")
        print(f"  Physical cores detected: {core_info.get('physical_cores', 'N/A')}")
        print("  Note: P-Core/E-Core type detection is not implemented.")
    else:
        print("  Could not retrieve core information.")
    print("-----------------------------\n")

    if isinstance(core_info.get("logical_cores"), int) and core_info["logical_cores"] > 0 :
        perform_individual_core_tests(core_info, test_duration_per_core)
    else:
        print("Skipping individual core tests as valid core information is unavailable.", file=sys.stderr)
        print("Running benchmarks on unspecified core(s) as a fallback general test:")
        # Fallback: Run benchmarks without setting affinity if core info is bad
        fallback_duration = 2
        print(f"  Running integer benchmark (fallback) for {fallback_duration}s...")
        int_ops = run_integer_benchmark(fallback_duration)
        int_ops_per_sec = int_ops / fallback_duration if fallback_duration > 0 else 0
        print(f"  Integer Ops/sec (fallback): {int_ops_per_sec:,.2f}")

        print(f"  Running float benchmark (fallback) for {fallback_duration}s...")
        float_ops = run_float_benchmark(fallback_duration)
        float_ops_per_sec = float_ops / fallback_duration if fallback_duration > 0 else 0
        print(f"  Float Ops/sec (fallback): {float_ops_per_sec:,.2f}")


    print("\nCPU Benchmark Suite Finished.")


# --- Group Performance Tests ---
def perform_group_test(core_info: dict, test_duration_sec: int, use_logical_cores: bool, benchmark_to_run: str):
    """
    Performs integer and/or float benchmarks concurrently on a group of cores.
    'benchmark_to_run' can be "all", "integer", or "float".
    """
    group_name = "All Logical Cores" if use_logical_cores else "All Physical Cores"
    print(f"\n==== {group_name} Performance Test (Duration: {test_duration_sec}s, Type: {benchmark_to_run}) ====")

    if use_logical_cores:
        num_cores_to_test = core_info.get('logical_cores')
    else:
        num_cores_to_test = core_info.get('physical_cores')

    if not isinstance(num_cores_to_test, int) or num_cores_to_test <= 0:
        print(f"Error: Valid number of cores for '{group_name}' not found ({num_cores_to_test}). Skipping this group test.", file=sys.stderr)
        if psutil is None and use_logical_cores: # Physical core count often relies on psutil
             print("Info: psutil is not installed, which might be why core count is unavailable or limited.", file=sys.stderr)
        return

    if benchmark_to_run == "all" or benchmark_to_run == "integer":
        # --- Integer Operations Group Test ---
        print(f"\nStarting Integer tests for {group_name} ({num_cores_to_test} cores, {test_duration_sec}s each thread)...")
        results_int = []
        threads_int = []
        for i in range(num_cores_to_test):
            cpu_to_pin = i # Pins to logical cores 0..N-1
            thread = threading.Thread(target=benchmark_thread_target, args=(cpu_to_pin, "integer", test_duration_sec, results_int))
            threads_int.append(thread)
            thread.start()

        for thread in threads_int:
            thread.join()

        total_int_ops = 0
        successful_int_threads = 0
        print(f"  Integer Test Results ({group_name}):")
        for res in results_int:
            if len(res) == 4 and res[3] == "Affinity Error": # Error case
                print(f"    CPU {res[0]}: Error - {res[3]}. Operations: {res[1]}")
            elif len(res) == 3: # Success case
                ops_per_sec = res[1] / test_duration_sec if test_duration_sec > 0 else 0
                print(f"    CPU {res[0]}: {res[1]} ops ({ops_per_sec:,.2f} Ops/sec)")
                total_int_ops += res[1]
                successful_int_threads +=1
            else: # Unknown error or format
                print(f"    CPU {res[0]}: Malformed result - {res}")


        if successful_int_threads > 0 :
            avg_int_ops_per_sec_per_thread = (total_int_ops / successful_int_threads) / test_duration_sec if test_duration_sec > 0 else 0
            total_int_ops_per_sec_aggregate = total_int_ops / test_duration_sec if test_duration_sec > 0 else 0
            print(f"  Total Integer Ops ({group_name}, {successful_int_threads} threads): {total_int_ops}")
            print(f"  Aggregate Integer Ops/sec ({group_name}): {total_int_ops_per_sec_aggregate:,.2f}")
            print(f"  Average Integer Ops/sec per thread ({group_name}): {avg_int_ops_per_sec_per_thread:,.2f}")
        else:
            print(f"  No successful integer benchmark threads completed for {group_name}.")

    if benchmark_to_run == "all" or benchmark_to_run == "float":
        # --- Floating-Point Operations Group Test ---
        print(f"\nStarting Floating-Point tests for {group_name} ({num_cores_to_test} cores, {test_duration_sec}s each thread)...")
        results_float = []
        threads_float = []
        for i in range(num_cores_to_test):
            cpu_to_pin = i
            thread = threading.Thread(target=benchmark_thread_target, args=(cpu_to_pin, "float", test_duration_sec, results_float))
            threads_float.append(thread)
            thread.start()

        for thread in threads_float:
            thread.join()

        total_float_ops = 0
        successful_float_threads = 0
        print(f"  Floating-Point Test Results ({group_name}):")
        for res in results_float:
            if len(res) == 4 and res[3] == "Affinity Error":
                print(f"    CPU {res[0]}: Error - {res[3]}. Operations: {res[1]}")
            elif len(res) == 3:
                ops_per_sec = res[1] / test_duration_sec if test_duration_sec > 0 else 0
                print(f"    CPU {res[0]}: {res[1]} ops ({ops_per_sec:,.2f} Ops/sec)")
                total_float_ops += res[1]
                successful_float_threads +=1
            else:
                 print(f"    CPU {res[0]}: Malformed result - {res}")


        if successful_float_threads > 0:
            avg_float_ops_per_sec_per_thread = (total_float_ops / successful_float_threads) / test_duration_sec if test_duration_sec > 0 else 0
            total_float_ops_per_sec_aggregate = total_float_ops / test_duration_sec if test_duration_sec > 0 else 0
            print(f"  Total Float Ops ({group_name}, {successful_float_threads} threads): {total_float_ops}")
            print(f"  Aggregate Float Ops/sec ({group_name}): {total_float_ops_per_sec_aggregate:,.2f}")
            print(f"  Average Float Ops/sec per thread ({group_name}): {avg_float_ops_per_sec_per_thread:,.2f}")
        else:
            print(f"  No successful float benchmark threads completed for {group_name}.")

    print(f"==== {group_name} Performance Test Finished ====")


def parse_cpu_arguments(): # Definition of parse_cpu_arguments
    """
    Parses command-line arguments for the CPU benchmark tool.
    """
    epilog_text = """
Examples:
  %(prog)s --run_mode individual --duration_individual 30
  %(prog)s --run_mode logical --duration_group 60 --test_type float
  %(prog)s --run_mode all --duration_individual 10 --duration_group 30
  %(prog)s # Runs all tests with default durations (10s individual, 15s group)
"""
    parser = argparse.ArgumentParser(
        description="CPU Benchmark Tool to measure integer and float performance across various core configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_text
    )
    parser.add_argument(
        "--test_type", type=str, default="all", choices=["all", "integer", "float"],
        help="Specifies which type of benchmark calculations to run (default: all)."
    )
    parser.add_argument(
        "--run_mode", type=str, default="all", choices=["all", "individual", "physical", "logical"],
        help="Specifies which test configurations to run: "
             "'individual' (each logical core separately), "
             "'physical' (group test on all physical cores), "
             "'logical' (group test on all logical cores), "
             "'all' (all modes). Default: all."
    )
    parser.add_argument(
        "--duration_individual", type=int, default=5,
        help="Duration in seconds for individual core tests (default: 2)."
    )
    parser.add_argument(
        "--duration_group", type=int, default=15,
        help="Duration in seconds for group (all physical/all logical) tests (default: 15)."
    )
    return parser.parse_args()

def perform_single_thread_cputest():
    
    print("Running one thread CPU benchmarks...\n")

    # Integer Benchmark
    int_duration = 8.0
    print(f"Starting integer benchmark for {int_duration} seconds...")
    int_ops = run_integer_benchmark(int_duration)
    int_ops_per_sec = int_ops / int_duration
    print(f"Integer benchmark completed: {int_ops} operations")
    print(f"Integer operations per second: {int_ops_per_sec:,.2f}\n")

    # Floating-Point Benchmark
    float_duration = 8.0
    print(f"Starting floating-point benchmark for {float_duration} seconds...")
    float_ops = run_float_benchmark(float_duration)
    float_ops_per_sec = float_ops / float_duration
    print(f"Floating-point benchmark completed: {float_ops} operations")
    print(f"Floating-point operations per second: {float_ops_per_sec:,.2f}\n")

    print("CPU benchmarks finished.\n")

    # Core Information
    print("Detecting CPU core information...")
    core_info = get_core_info()
    if core_info:
        print(f"Logical cores: {core_info.get('logical_cores', 'N/A')}")
        print(f"Physical cores: {core_info.get('physical_cores', 'N/A')}")
    else:
        print("Could not retrieve core information.")

    print("\nAll benchmarks and checks complete.")



def main():
    """
    Main function to orchestrate CPU benchmarks.
    """
    args = parse_cpu_arguments()

    print(f"Starting CPU Benchmark Suite with arguments: {vars(args)}\n") # Using vars(args) for cleaner print

    core_info = get_core_info()
    print("--- System Core Information ---")
    if core_info:
        print(f"  Logical cores detected: {core_info.get('logical_cores', 'N/A')}")
        print(f"  Physical cores detected: {core_info.get('physical_cores', 'N/A')}")
    else:
        print("  Could not retrieve core information.")
    print("-----------------------------\n")

    valid_logical_cores = isinstance(core_info.get("logical_cores"), int) and core_info["logical_cores"] > 0
    valid_physical_cores = isinstance(core_info.get("physical_cores"), int) and core_info["physical_cores"] > 0

    # Determine if any test mode involving cores is selected
    core_tests_selected = args.run_mode in ["all", "individual", "physical", "logical"]

    if core_tests_selected and not valid_logical_cores:
        print("Warning: Core-specific run modes selected, but valid logical core information is unavailable. Attempting fallback tests.", file=sys.stderr)

    if args.run_mode == "all" or args.run_mode == "individual":
        if valid_logical_cores:
            perform_individual_core_tests(core_info, args.duration_individual, args.test_type)
        elif not core_tests_selected: # Only print if individual was the *only* mode or not part of 'all'
             print("Skipping individual core tests as valid logical core information is unavailable.", file=sys.stderr)


    physical_group_test_run_as_logical_fallback = False
    if args.run_mode == "all" or args.run_mode == "physical":
        if valid_physical_cores:
            perform_group_test(core_info, args.duration_group, use_logical_cores=False, benchmark_to_run=args.test_type)
        elif valid_logical_cores : # Fallback to logical if physical count is bad but logical is good
             print("\nWarning: Valid physical core count not available. Running 'All Physical Cores' test on all logical cores instead.", file=sys.stderr)
             perform_group_test(core_info, args.duration_group, use_logical_cores=True, benchmark_to_run=args.test_type)
             physical_group_test_run_as_logical_fallback = True # Mark that logical cores test was used for physical
        elif not core_tests_selected:
            print("\nInfo: Valid physical core count not available. Skipping 'All Physical Cores' group test.", file=sys.stderr)

    if args.run_mode == "all" or args.run_mode == "logical":
        if valid_logical_cores:
            run_logical_group_test = True
            if physical_group_test_run_as_logical_fallback:
                 print("\nInfo: 'All Logical Cores' group test was already run as a fallback for 'All Physical Cores'. Skipping redundant run.")
                 run_logical_group_test = False
            elif valid_physical_cores and core_info.get('physical_cores') == core_info.get('logical_cores'):
                 print("\nInfo: Physical core count is the same as logical core count. 'All Logical Cores' test results would be redundant with 'All Physical Cores' (if run). Skipping.")
                 # This check might need refinement if 'physical' mode wasn't part of 'all'
                 if not (args.run_mode == "all" or args.run_mode == "physical"): # If physical wasn't run, then logical is not redundant
                     pass # Allow it
                 else: # Physical was run or would have been
                     run_logical_group_test = False

            if run_logical_group_test:
                 perform_group_test(core_info, args.duration_group, use_logical_cores=True, benchmark_to_run=args.test_type)
        elif not core_tests_selected:
            print("\nInfo: Valid logical core count not available. Skipping 'All Logical Cores' group test.", file=sys.stderr)

    # Fallback for when no specific core tests are run due to unavailable core info OR if no valid run_mode was effectively chosen
    # This condition needs to be more robust: e.g. if user chose 'individual' but logical_cores is bad.
    tests_actually_run_count = 0
    if (args.run_mode == "all" or args.run_mode == "individual") and valid_logical_cores: tests_actually_run_count+=1
    if (args.run_mode == "all" or args.run_mode == "physical") and (valid_physical_cores or (valid_logical_cores and not valid_physical_cores)): tests_actually_run_count+=1 # counts physical or its logical fallback
    # counting logical group test is tricky due to redundancy skips
    # A simpler check: if no tests were dispatched above and some core-based mode was intended.

    """     # Simplified Fallback: If 'all' modes were intended but logical cores are bad, then nothing ran.
    if not valid_logical_cores and args.run_mode != "none": # (assuming "none" is not a choice yet)
        print("\nWarning: No specific core-based tests could be run due to unavailable core information.", file=sys.stderr)
        print("Running general fallback benchmarks (unpinned):")

        fallback_duration = args.duration_individual # Use individual duration as a fallback
        if args.test_type == "all" or args.test_type == "integer":
            print(f"  Running integer benchmark (fallback) for {fallback_duration}s...")
            int_ops = run_integer_benchmark(fallback_duration)
            int_ops_per_sec = int_ops / fallback_duration if fallback_duration > 0 else 0
            print(f"  Integer Ops/sec (fallback): {int_ops_per_sec:,.2f}")

        if args.test_type == "all" or args.test_type == "float":
            print(f"  Running float benchmark (fallback) for {fallback_duration}s...")
            float_ops = run_float_benchmark(fallback_duration)
            float_ops_per_sec = float_ops / fallback_duration if fallback_duration > 0 else 0
            print(f"  Float Ops/sec (fallback): {float_ops_per_sec:,.2f}")
    """

    print("\nCPU Benchmark Suite Finished.")


if __name__ == "__main__":
    main()