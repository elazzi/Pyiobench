import os
import sys
import time
import argparse
import subprocess
import random

def human_size_to_bytes(size_str):
    """
    Converts human-readable size string (e.g., "1G", "500M") to bytes.
    """
    size_str = size_str.strip().upper()
    if size_str.endswith("G"):
        return int(float(size_str[:-1]) * 1024**3)
    elif size_str.endswith("M"):
        return int(float(size_str[:-1]) * 1024**2)
    elif size_str.endswith("K"):
        return int(float(size_str[:-1]) * 1024)
    elif size_str.endswith("B"):
        return int(float(size_str[:-1]))
    else:
        try:
            return int(float(size_str)) # Assume bytes if no suffix
        except ValueError:
            raise ValueError(f"Invalid size string: {size_str}")


DEFAULT_BLOCK_SIZES_KB = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 4*1024, 8*1024] # In KB
DEFAULT_BLOCK_SIZES_BYTES = [s * 1024 for s in DEFAULT_BLOCK_SIZES_KB]

def detect_filesystem_info(target_directory: str):
    """
    Detects filesystem type and block size for the given directory.
    Supports Linux/macOS (using df/stat) and Windows (using ctypes).
    """
    if os.name == 'nt':
        # Windows-specific implementation
        try:
            import ctypes
            from ctypes import wintypes

            fs_type_str = "Error"
            block_size_val = "Error"

            # Adjust path for root drives (e.g., "C:" to "C:\\")
            # GetDiskFreeSpaceW and GetVolumeInformationW require a trailing backslash for root paths.
            adjusted_target_directory = target_directory
            if len(target_directory) == 2 and target_directory[1] == ':': # e.g. "C:"
                adjusted_target_directory = target_directory + "\\"
            # For paths like "C:\mydir", it should already be fine.
            # os.path.join(target_directory, '') might also work but direct manipulation is clearer here for root.


            # Get Block Size
            sectors_per_cluster = wintypes.DWORD()
            bytes_per_sector = wintypes.DWORD()
            number_of_free_clusters = wintypes.DWORD()
            total_number_of_clusters = wintypes.DWORD()

            success_disk_space = ctypes.windll.kernel32.GetDiskFreeSpaceW(
                ctypes.c_wchar_p(adjusted_target_directory),
                ctypes.byref(sectors_per_cluster),
                ctypes.byref(bytes_per_sector),
                ctypes.byref(number_of_free_clusters),
                ctypes.byref(total_number_of_clusters)
            )
            if success_disk_space:
                block_size_val = sectors_per_cluster.value * bytes_per_sector.value
            else:
                error_code = ctypes.get_last_error()
                print(f"Windows API Error (GetDiskFreeSpaceW) for '{adjusted_target_directory}': {error_code}", file=sys.stderr)

            # Get Filesystem Type
            fs_name_buffer = ctypes.create_unicode_buffer(1024)
            # Other parameters for GetVolumeInformationW that we don't necessarily need for fs_type
            volume_name_buffer = ctypes.create_unicode_buffer(1024)
            volume_serial_number = wintypes.DWORD()
            maximum_component_length = wintypes.DWORD()
            file_system_flags = wintypes.DWORD()

            success_volume_info = ctypes.windll.kernel32.GetVolumeInformationW(
                ctypes.c_wchar_p(adjusted_target_directory), # Path to the root of the volume
                volume_name_buffer, ctypes.sizeof(volume_name_buffer),
                ctypes.byref(volume_serial_number),
                ctypes.byref(maximum_component_length),
                ctypes.byref(file_system_flags),
                fs_name_buffer, ctypes.sizeof(fs_name_buffer)
            )
            if success_volume_info:
                fs_type_str = fs_name_buffer.value
            else:
                error_code = ctypes.get_last_error()
                print(f"Windows API Error (GetVolumeInformationW) for '{adjusted_target_directory}': {error_code}", file=sys.stderr)
                # fs_type_str remains "Error" if this fails

            return {"type": fs_type_str, "block_size": block_size_val}

        except ImportError:
            print("Error: ctypes or wintypes module not found. Windows API calls cannot be made.", file=sys.stderr)
            return {"type": "Error", "block_size": "Error"}
        except AttributeError:
            print("Error: `ctypes.windll.kernel32` not available. Ensure you are on a Windows system.", file=sys.stderr)
            return {"type": "Error", "block_size": "Error"}
        except Exception as e:
            print(f"An unexpected error occurred during Windows filesystem info detection: {e}", file=sys.stderr)
            return {"type": "Error", "block_size": "Error"}

    else:
        # Existing Linux/macOS implementation
        try:
            # Get filesystem type
            df_process = subprocess.run(['df', '-T', target_directory], capture_output=True, text=True, check=True)
            df_output = df_process.stdout.strip().split('\n')
            if len(df_output) > 1:
                fs_type = df_output[1].split()[1]
            else:
                fs_type = "Unknown"

            # Get block size using stat on the directory itself
            # For Linux, `stat -f -c %S target_directory` gives fundamental block size of the filesystem.
            # `stat -c %B target_directory` gives the size in bytes of each block reported by `stat -c %b`.
            # Let's use %S for fundamental block size.
            stat_process = subprocess.run(['stat', '-f', '-c', '%S', target_directory], capture_output=True, text=True, check=True)
            block_size = int(stat_process.stdout.strip())

            return {"type": fs_type, "block_size": block_size}
        except FileNotFoundError as e:
            print(f"Error: Required command ('df' or 'stat') not found. Please ensure you are on a Linux-like system and these commands are in your PATH. ({e})", file=sys.stderr)
            return {"type": "Error", "block_size": "Error"}
        except subprocess.CalledProcessError as e:
            print(f"Error executing filesystem command ('df' or 'stat'): {e}", file=sys.stderr)
            return {"type": "Error", "block_size": "Error"}
        except Exception as e:
            print(f"An unexpected error occurred while detecting filesystem info on non-Windows OS: {e}", file=sys.stderr)
            return {"type": "Error", "block_size": "Error"}

def benchmark_sequential_writes(directory, file_size_str, duration_sec, block_size_bytes):
    """
    Benchmarks sequential writes to a test file.
    """
    print(f"\nStarting sequential write benchmark...")
    file_size_bytes = human_size_to_bytes(file_size_str)
    test_file_path = os.path.join(directory, "sequential_test_file.dat")
    bytes_written = 0
    start_time = time.monotonic()
    operations = 0

    try:
        with open(test_file_path, "wb") as f:
            while time.monotonic() - start_time < duration_sec:
                # Ensure we don't write more than file_size_bytes if duration is very long
                # or block_size is very small, by writing in chunks up to file_size_bytes
                # and then seeking back to the beginning if the file is filled.
                # This is a common approach for continuous benchmarking over a fixed duration.
                if f.tell() + block_size_bytes > file_size_bytes:
                    f.seek(0) # Loop within the file if duration not met

                data_chunk = os.urandom(block_size_bytes)
                f.write(data_chunk)
                f.flush() # Ensure data is written to disk, not just buffered
                os.fsync(f.fileno()) # Force write to disk
                bytes_written += len(data_chunk)
                operations += 1
                if time.monotonic() - start_time >= duration_sec:
                    break
        end_time = time.monotonic()
    except IOError as e:
        print(f"Error during sequential write operation for file {test_file_path}: {e}", file=sys.stderr)
        return {"bytes_written": 0, "time_taken": 0, "speed_mbps": 0, "iops": 0, "error": str(e)}
    finally:
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

    time_taken = end_time - start_time
    speed_mbps = (bytes_written / (1024**2)) / time_taken if time_taken > 0 else 0
    iops = operations / time_taken if time_taken > 0 else 0

    print(f"Sequential write: {bytes_written / (1024**2):.2f} MB written in {time_taken:.2f}s")
    print(f"Speed: {speed_mbps:.2f} MB/s, IOPS: {iops:.2f}")
    return {"bytes_written": bytes_written, "time_taken": time_taken, "speed_mbps": speed_mbps, "iops": iops}


def benchmark_random_writes(directory, file_size_str, duration_sec, block_size_bytes):
    """
    Benchmarks random writes to a test file.
    """
    print(f"\nStarting random write benchmark...")
    file_size_bytes = human_size_to_bytes(file_size_str)
    test_file_path = os.path.join(directory, "random_test_file.dat")
    bytes_written = 0
    start_time = time.monotonic()
    operations = 0

    try:
        # Pre-allocate the file to its full size to ensure random writes are truly random within the boundary
        with open(test_file_path, "wb") as f:
            f.seek(file_size_bytes - 1)
            f.write(b'\0') # Write a single byte at the end to set the file size

        with open(test_file_path, "r+b") as f: # Open for read and write in binary mode
            while time.monotonic() - start_time < duration_sec:
                random_offset = random.randint(0, file_size_bytes - block_size_bytes)
                f.seek(random_offset)
                data_chunk = os.urandom(block_size_bytes)
                f.write(data_chunk)
                f.flush() # Ensure data is written to disk
                os.fsync(f.fileno()) # Force write to disk
                bytes_written += len(data_chunk)
                operations += 1
                if time.monotonic() - start_time >= duration_sec:
                    break
        end_time = time.monotonic()
    except IOError as e:
        print(f"Error during random write operation for file {test_file_path}: {e}", file=sys.stderr)
        return {"bytes_written": 0, "time_taken": 0, "speed_mbps": 0, "iops": 0, "error": str(e)}
    except ValueError as e: # Handles potential error from random.randint if file_size_bytes < block_size_bytes
        print(f"Error setting up random write for file {test_file_path} (check file size vs block size): {e}", file=sys.stderr)
        return {"bytes_written": 0, "time_taken": 0, "speed_mbps": 0, "iops": 0, "error": str(e)}
    finally:
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

    time_taken = end_time - start_time
    speed_mbps = (bytes_written / (1024**2)) / time_taken if time_taken > 0 else 0
    iops = operations / time_taken if time_taken > 0 else 0

    print(f"Random write: {bytes_written / (1024**2):.2f} MB written in {time_taken:.2f}s")
    print(f"Speed: {speed_mbps:.2f} MB/s, IOPS: {iops:.2f}")
    return {"bytes_written": bytes_written, "time_taken": time_taken, "speed_mbps": speed_mbps, "iops": iops}


def parse_arguments():
    """
    Parses command-line arguments for the disk benchmark tool.
    """
    epilog = """
Examples:
  %(prog)s -d /mnt/my_drive -s 2G -r 60
  %(prog)s --block_sizes 4k,16k,1M
  %(prog)s -d . -s 500M --duration 20 --block_sizes 8192,16384

Note: Filesystem type and block size detection is supported on Linux, macOS, and Windows.
      Full filesystem type details are typically more comprehensive on Linux/macOS.
"""
    parser = argparse.ArgumentParser(
        description="A Python script to benchmark disk write performance (sequential and random) and identify optimal block sizes. Detects filesystem type and block size on Linux, macOS (via system commands), and Windows (via Windows API).",
        formatter_class=argparse.RawDescriptionHelpFormatter, # To ensure epilog formatting is preserved
        epilog=epilog
    )
    parser.add_argument("-d", "--directory", type=str, default=".",
                        help="Directory to run the benchmark in (default: current directory)")
    parser.add_argument("-s", "--size", type=str, default="1G",
                        help="Size of the test file (e.g., 1G, 500M, default: 1G)")
    parser.add_argument("-r", "--duration", type=int, default=30,
                        help="Duration of each test in seconds (default: 30)")
    parser.add_argument("--block_sizes", type=str,
                        help="Comma-separated list of block sizes to test (e.g., \"4k,1M,8M\"). "
                             "Overrides detected block size and default list. Supports suffixes like K, M, G.")
    # Add more arguments as needed, e.g., for specific tests, block sizes
    return parser.parse_args()

def print_results(fs_info, all_seq_results, all_rand_results, best_seq_result, best_rand_result, args): # Renamed parameters
    """
    Prints benchmarking results in a formatted way.
    """
    print("\n\n--- Overall Benchmark Summary ---")
    print(f"Test Directory: {os.path.abspath(args.directory)}")
    print(f"Test File Size: {args.size} ({human_size_to_bytes(args.size)} bytes)")
    print(f"Test Duration per run: {args.duration}s")
    print(f"Filesystem Type: {fs_info.get('type', 'N/A')}")
    native_fs_block_size = fs_info.get('block_size', 'N/A') # Renamed for clarity
    if isinstance(native_fs_block_size, int):
        print(f"Native Filesystem Block Size: {native_fs_block_size // 1024}K ({native_fs_block_size} bytes)")
    else:
        print(f"Native Filesystem Block Size: {native_fs_block_size}")


    # Optional: Print detailed results for all block sizes
    # print_detailed_results = True # Or make this a command line arg
    # if print_detailed_results:
    print("\n--- Detailed Results per Block Size ---")

    print("\nSequential Write Tests:")
    if all_seq_results:
        for bs, result in sorted(all_seq_results.items()):
            bs_kb = bs // 1024
            if "error" in result:
                print(f"  Block Size: {bs_kb}K ({bs} bytes) - Error: {result['error']}")
            else:
                print(f"  Block Size: {bs_kb}K ({bs} bytes): Speed: {result.get('speed_mbps', 0):.2f} MB/s, IOPS: {result.get('iops', 0):.2f}")
    else:
        print("  No results available for sequential write tests.")

    print("\nRandom Write Tests:")
    if all_rand_results:
        for bs, result in sorted(all_rand_results.items()):
            bs_kb = bs // 1024
            if "error" in result:
                 print(f"  Block Size: {bs_kb}K ({bs} bytes) - Error: {result['error']}")
            # Check if 'bytes_written' is present and is 0, and no error, to identify skipped tests
            elif result.get('bytes_written', -1) == 0 and 'error' not in result:
                 print(f"  Block Size: {bs_kb}K ({bs} bytes) - Skipped or no data written (e.g. file size < block size).")
            else:
                print(f"  Block Size: {bs_kb}K ({bs} bytes): Speed: {result.get('speed_mbps', 0):.2f} MB/s, IOPS: {result.get('iops', 0):.2f}")
    else:
        print("  No results available for random write tests.")

    print("\n--- Best Performers ---")
    if best_seq_result and 'block_size' in best_seq_result:
        bs_kb = best_seq_result['block_size'] // 1024
        print(f"Best Sequential Write Performance with Block Size: {bs_kb}K ({best_seq_result['block_size']} bytes)")
        print(f"  Speed: {best_seq_result.get('speed_mbps',0):.2f} MB/s")
        print(f"  IOPS: {best_seq_result.get('iops',0):.2f} operations/sec")
        print(f"  Total Data Written: {best_seq_result.get('bytes_written',0) / (1024**2):.2f} MB in {best_seq_result.get('time_taken',0):.2f}s")
    else:
        print("Sequential write tests did not yield a best performer (or all failed).")

    if best_rand_result and 'block_size' in best_rand_result:
        bs_kb = best_rand_result['block_size'] // 1024
        print(f"Best Random Write Performance with Block Size: {bs_kb}K ({best_rand_result['block_size']} bytes)")
        print(f"  Speed: {best_rand_result.get('speed_mbps',0):.2f} MB/s")
        print(f"  IOPS: {best_rand_result.get('iops',0):.2f} operations/sec")
        print(f"  Total Data Written: {best_rand_result.get('bytes_written',0) / (1024**2):.2f} MB in {best_rand_result.get('time_taken',0):.2f}s")
    else:
        print("Random write tests did not yield a best performer (or all failed).")

    print("\n--- End of Summary ---")

def main():
    """
    Main function to orchestrate the benchmarking process.
    """
    args = parse_arguments()

    # --- Pre-benchmark checks for target directory ---
    abs_directory_path = os.path.abspath(args.directory)
    if not os.path.exists(abs_directory_path):
        print(f"Error: Target directory '{abs_directory_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(abs_directory_path):
        print(f"Error: Target path '{abs_directory_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    if not os.access(abs_directory_path, os.W_OK):
        print(f"Error: Target directory '{abs_directory_path}' is not writable.", file=sys.stderr)
        sys.exit(1)
    # --- End pre-benchmark checks ---

    filesystem_info = detect_filesystem_info(abs_directory_path) # Use absolute path
    native_block_size = filesystem_info.get("block_size") if isinstance(filesystem_info.get("block_size"), int) else None

    if args.block_sizes:
        try:
            test_block_sizes = [human_size_to_bytes(s.strip()) for s in args.block_sizes.split(',')]
            if not test_block_sizes: # Handle empty string case
                 raise ValueError("Block size list cannot be empty.")
        except ValueError as e:
            print(f"Error parsing custom block sizes ('{args.block_sizes}'): {e}", file=sys.stderr)
            sys.exit(1)
    else:
        test_block_sizes = list(DEFAULT_BLOCK_SIZES_BYTES) # Use a copy
        if native_block_size and native_block_size not in test_block_sizes:
            test_block_sizes.append(native_block_size)
            test_block_sizes.sort()

    print(f"Target directory: {abs_directory_path}") # Use absolute path
    print(f"Test file size: {args.size}")
    print(f"Test duration per run: {args.duration}s")
    if native_block_size:
        print(f"Native filesystem block size: {native_block_size // 1024}K ({native_block_size} bytes)")
    else:
        print(f"Native filesystem block size: Could not be determined.")
    print(f"Testing with block sizes (bytes): {test_block_sizes}")
    print(f"Testing with block sizes (KB): {[str(bs // 1024) + 'K' for bs in test_block_sizes]}")


    all_sequential_results = {}
    all_random_results = {}

    best_seq_result = None
    best_rand_result = None

    file_size_bytes = human_size_to_bytes(args.size)

    for block_size_bytes_iter in test_block_sizes:
        current_block_size_kb = block_size_bytes_iter // 1024
        print(f"\n--- Testing with block size: {current_block_size_kb}K ({block_size_bytes_iter} bytes) ---")

        # Sequential Write Test
        try:
            print(f"Running sequential write test with block size {current_block_size_kb}K...")
            seq_result = benchmark_sequential_writes(abs_directory_path, args.size, args.duration, block_size_bytes_iter) # Use absolute path
            all_sequential_results[block_size_bytes_iter] = seq_result
            if seq_result and seq_result.get('speed_mbps', 0) > (best_seq_result.get('speed_mbps', 0) if best_seq_result else -1):
                best_seq_result = seq_result
                best_seq_result['block_size'] = block_size_bytes_iter # Store block_size with the result
        except Exception as e:
            print(f"Error during sequential write test with block size {current_block_size_kb}K: {e}", file=sys.stderr)
            all_sequential_results[block_size_bytes_iter] = {"error": str(e), "speed_mbps": 0, "iops": 0}


        # Random Write Test
        if file_size_bytes < block_size_bytes_iter:
            print(f"Skipping random write test for block size {current_block_size_kb}K: test file size ({args.size} = {file_size_bytes} bytes) is smaller than block size ({block_size_bytes_iter} bytes).")
            all_random_results[block_size_bytes_iter] = {"bytes_written": 0, "time_taken": 0, "speed_mbps": 0, "iops": 0, "error": "File size smaller than block size"}
            continue

        try:
            print(f"Running random write test with block size {current_block_size_kb}K...")
            rand_result = benchmark_random_writes(abs_directory_path, args.size, args.duration, block_size_bytes_iter) # Use absolute path
            all_random_results[block_size_bytes_iter] = rand_result
            if rand_result and rand_result.get('speed_mbps', 0) > (best_rand_result.get('speed_mbps', 0) if best_rand_result else -1):
                best_rand_result = rand_result
                best_rand_result['block_size'] = block_size_bytes_iter # Store block_size with the result
        except Exception as e:
            print(f"Error during random write test with block size {current_block_size_kb}K: {e}", file=sys.stderr)
            all_random_results[block_size_bytes_iter] = {"error": str(e), "speed_mbps": 0, "iops": 0}

    print_results(filesystem_info, all_sequential_results, all_random_results, best_seq_result, best_rand_result, args)

if __name__ == "__main__":
    main()
