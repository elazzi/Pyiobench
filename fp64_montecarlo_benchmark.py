"""
FP64 Monte Carlo CPU/GPU Benchmark Script
... (module docstring - assuming it's largely the same)
"""

import time
import concurrent.futures
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import numpy as np
except ImportError:
    logging.error("NumPy library not found. This script requires NumPy. Please install it (e.g., pip install numpy).")
    exit(1)

PYOPENCL_AVAILABLE = False
cl = None
cl_array = None
cl_random = None
try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    import pyopencl.clrandom as cl_random
    PYOPENCL_AVAILABLE = True
    logging.info("PyOpenCL and pyopencl.clrandom found. GPU backend will be available if selected.")
except ImportError:
    logging.info("PyOpenCL or pyopencl.clrandom not found. GPU backend will not be available.")

OPENCL_KERNEL_CODE = """
// ... (kernel code unchanged)
__kernel void monte_carlo_ops_kernel(
    __global const double* random_numbers, __global double* output_buffer, const int data_size)
{ int gid = get_global_id(0); if (gid >= data_size) { return; } double r = random_numbers[gid];
  double v1 = log(r + 1.0e-9); double v2 = exp(v1 / 2.0); double v3 = sqrt(v2 * 0.5);
  double v4 = pow(v3, 1.5); double v5 = v1 + v2 - v3 * v4; output_buffer[gid] = v5; }
"""

FLOPS_PER_ELEMENT_IN_STEP = 10

def get_pyopencl_context_queue(platform_name_filter: str = None, device_name_filter: str = None) -> tuple[any, any, str, str]:
    """Initializes and returns a PyOpenCL context, command queue, platform name, and device name."""
    # This function's existing try-except cl.Error and general Exception handling is deemed sufficient.
    # It already logs errors and returns None, None, None, None on failure.
    if not PYOPENCL_AVAILABLE:
        logging.error("get_pyopencl_context_queue called but PyOpenCL is not available.")
        return None, None, None, None
    logging.info("Starting OpenCL device selection...")
    try:
        platforms = cl.get_platforms()
        if not platforms: logging.error("No OpenCL platforms found."); return None, None, None, None
        target_platform = None
        if platform_name_filter:
            for p in platforms:
                if platform_name_filter.lower() in p.name.lower(): target_platform = p; break
            if not target_platform: logging.error(f"No platform matching '{platform_name_filter}'."); return None, None, None, None
        elif len(platforms) == 1: target_platform = platforms[0]
        else: logging.info(f"Multiple platforms, using first: {platforms[0].name}."); target_platform = platforms[0]
        platform_name_out = target_platform.name.strip()
        logging.info(f"Selected Platform: {platform_name_out} ({target_platform.version.strip()})")
        devices = target_platform.get_devices()
        if not devices: logging.error(f"No devices on platform '{platform_name_out}'."); return None, None, None, None
        target_device = None
        if device_name_filter:
            for d in devices:
                if device_name_filter.lower() in d.name.lower(): target_device = d; break
            if not target_device: logging.error(f"No device on '{platform_name_out}' matching '{device_name_filter}'."); return None, None, None, None
        elif len(devices) == 1: target_device = devices[0]
        else:
            gpu_devices = [d for d in devices if d.type == cl.device_type.GPU]
            if gpu_devices: target_device = gpu_devices[0]; logging.info(f"Multiple GPUs on '{platform_name_out}', using first: {target_device.name}.")
            else: target_device = devices[0]; logging.warning(f"No GPU on '{platform_name_out}', using first device: {target_device.name}.")
        device_name_out = target_device.name.strip()
        logging.info(f"Selected Device: {device_name_out}")
        context = cl.Context([target_device])
        queue = cl.CommandQueue(context)
        logging.info("OpenCL context and queue created.")
        return context, queue, platform_name_out, device_name_out
    except cl.Error as e: logging.error(f"OpenCL Error during setup: {e.what()} (Code: {e.code})", exc_info=True); return None, None, None, None
    except Exception as e: logging.error(f"Unexpected error during OpenCL setup: {e}", exc_info=True); return None, None, None, None

def monte_carlo_simulation_step(data_size: int) -> float:
  """CPU version."""
  # ... (implementation unchanged)
  try:
    random_numbers = np.random.rand(data_size); v1 = np.log(random_numbers + 1e-9); v2 = np.exp(v1 / 2.0); v3 = np.sqrt(v2 * 0.5); v4 = np.power(v3, 1.5); v5 = v1 + v2 - v3 * v4; return float(np.sum(v5))
  except MemoryError: logging.error(f"CPU MemoryError (data_size {data_size}).", exc_info=True); raise
  except Exception as e: logging.error(f"CPU Exception: {e}", exc_info=True); raise

def monte_carlo_simulation_step_gpu(context, queue, kernel_program, data_size, cl_rng_generator) -> float:
    """Executes one step of the Monte Carlo simulation on the GPU."""
    if not PYOPENCL_AVAILABLE:
        raise RuntimeError("PyOpenCL is not available, cannot run GPU step.")
    try:
        # Buffer creation
        random_numbers_gpu = cl_array.empty(queue, data_size, dtype=np.float64)
        output_buffer_gpu = cl_array.empty(queue, data_size, dtype=np.float64)

        # Random number generation
        cl_rng_generator.fill_uniform(random_numbers_gpu, queue=queue)
        # queue.finish() # Optional: ensure RNG is done if fill_uniform is non-blocking & next step needs it immediately. Usually not an issue with subsequent kernel launch.

        # Kernel execution
        kernel_event = kernel_program.monte_carlo_ops_kernel(
            queue, (data_size,), None,
            random_numbers_gpu.data,
            output_buffer_gpu.data,
            np.int32(data_size)
        )
        kernel_event.wait() # Wait for the kernel event to complete

        # Result retrieval
        host_result_array = output_buffer_gpu.get(queue=queue) # This implicitly waits for copy

        result_sum = np.sum(host_result_array)
        return float(result_sum)

    except cl.Error as e:
        logging.error(f"A PyOpenCL error occurred during GPU simulation step: {e.what()} (Code: {e.code})", exc_info=True)
        # logging.error(f"  Kernel: {kernel_program}, Data Size: {data_size}") # Example of more context
        # logging.error(f"  Context: {context}, Queue: {queue.device.name}")
        raise
    except Exception as e: # Catch other non-OpenCL errors
        logging.error(f"A non-PyOpenCL error occurred during GPU simulation step: {e}", exc_info=True)
        raise

def worker_task(num_worker_paths: int, data_size_per_path: int) -> float:
  """CPU worker."""
  # ... (implementation unchanged)
  local_sum = 0.0
  for i in range(num_worker_paths):
    try: local_sum += monte_carlo_simulation_step(data_size_per_path)
    except Exception: logging.warning(f"Skipping CPU path {i+1}/{num_worker_paths} due to error.")
  return local_sum

def run_benchmark(num_paths: int, data_size_per_path: int, num_threads: int, backend: str,
                  cl_context=None, cl_queue=None) -> tuple[float, float]:
  """Main benchmark runner."""
  total_sum = 0.0

  if backend == "cpu":
    start_time = time.perf_counter()
    # ... (CPU logic unchanged)
    logging.info(f"Executing benchmark on CPU backend with {num_threads} thread(s).")
    if num_threads <= 1: total_sum = worker_task(num_paths, data_size_per_path)
    else:
      paths_per_thread_base = num_paths // num_threads; remainder_paths = num_paths % num_threads
      tasks_for_each_thread = []
      for i in range(num_threads):
          paths_for_this_thread = paths_per_thread_base
          if i < remainder_paths: paths_for_this_thread += 1
          if paths_for_this_thread > 0: tasks_for_each_thread.append(paths_for_this_thread)
      futures = []
      with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i, paths_count in enumerate(tasks_for_each_thread):
          logging.debug(f"Submitting CPU task: {paths_count} paths, thread {i+1}.")
          futures.append(executor.submit(worker_task, paths_count, data_size_per_path))
        for future in concurrent.futures.as_completed(futures):
          try: total_sum += future.result()
          except Exception as e: logging.error(f"CPU worker task exception: {e}", exc_info=True)
    end_time = time.perf_counter()

  elif backend == "gpu":
    logging.info("GPU backend selected.")
    if not cl_context or not cl_queue: raise ValueError("OpenCL context/queue required for GPU.")
    if not PYOPENCL_AVAILABLE or cl_random is None: raise RuntimeError("PyOpenCL/cl_random missing.")

    program = None
    cl_rng_generator = None
    try:
        logging.info("Compiling OpenCL kernel...")
        program = cl.Program(cl_context, OPENCL_KERNEL_CODE).build() # Build options can be added here if needed
        logging.info("OpenCL kernel compiled successfully.")
    except cl.Error as e:
        logging.error(f"OpenCL kernel compilation failed: {e.what()} (Code: {e.code})", exc_info=True)
        # Log detailed compiler output if available
        if hasattr(e, 'args') and len(e.args) > 1 and isinstance(e.args[1], list):
             for dev_log_pair in e.args[1]:
                 if len(dev_log_pair) == 2: logging.error(f"Device {dev_log_pair[0].name} compiler log:\n{dev_log_pair[1]}")
        raise

    try:
        current_seed = int(time.time())
        cl_rng_generator = cl_random.PhiloxGenerator(cl_context, seed=current_seed)
        logging.info(f"PyOpenCL RNG (Philox) initialized with seed: {current_seed}.")
    except cl.Error as e: # PhiloxGenerator can raise cl.Error, e.g. if context is invalid
        logging.error(f"Failed to initialize PyOpenCL RNG (Philox): {e.what()} (Code: {e.code})", exc_info=True)
        raise
    except Exception as e: # Other potential errors during RNG init
        logging.error(f"An unexpected error occurred during PyOpenCL RNG initialization: {e}", exc_info=True)
        raise

    start_time = time.perf_counter()
    logging.info(f"Starting {num_paths} simulation paths on GPU...")
    for i in range(num_paths):
        logging.debug(f"Executing GPU path {i+1}/{num_paths}...")
        try:
            total_sum += monte_carlo_simulation_step_gpu(cl_context, cl_queue, program, data_size_per_path, cl_rng_generator)
        except Exception as e:
            # Error already logged in monte_carlo_simulation_step_gpu
            logging.error(f"Stopping GPU benchmark after failure on path {i+1}/{num_paths}.")
            # Decide whether to return partial sum or indicate total failure.
            # For now, re-raise to stop entirely as one path failure might indicate systemic issue.
            raise
    end_time = time.perf_counter()
    logging.info("All GPU paths processed successfully.")
  else:
    logging.error(f"Unsupported backend: {backend}."); return 0.0, 0.0

  duration = end_time - start_time
  logging.info(f"Benchmark computation on {backend.upper()} completed. Sum: {total_sum:.4e}")
  return duration, total_sum

if __name__ == '__main__':
  # ... (ArgumentParser setup and args parsing unchanged)
  parser = argparse.ArgumentParser(description="FP64 Monte Carlo CPU/GPU Benchmark Script.",formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('--backend',type=str,default="cpu",choices=["cpu","gpu"],help="Execution backend.\n(default: cpu)")
  parser.add_argument('--num_paths',type=int,default=100,help="Number of simulation paths.\n(default: 100)")
  parser.add_argument('--data_size',type=int,default=100000,help="Data array size per path.\n(default: 100000)")
  parser.add_argument('--threads',type=int,default=1,help="Number of threads (CPU backend).\n(default: 1)")
  parser.add_argument("--gpu_platform_name",type=str,default=None,help="OpenCL platform name filter.\nOptional.")
  parser.add_argument("--gpu_device_name",type=str,default=None,help="OpenCL device name filter.\nOptional.")
  args = parser.parse_args()

  context, queue, selected_platform_name, selected_device_name = None, None, None, None

  if args.backend == "gpu":
      if not PYOPENCL_AVAILABLE: logging.error("GPU: PyOpenCL/clrandom not available."); exit(1)
      logging.info("Initializing OpenCL for GPU backend...")
      context, queue, selected_platform_name, selected_device_name = get_pyopencl_context_queue(args.gpu_platform_name, args.gpu_device_name)
      if not context or not queue: logging.error("GPU: Failed to init OpenCL context/queue. Exiting."); exit(1)

  # ... (logging and main try-except block unchanged)
  logging.info("="*50); logging.info("FP64 Monte Carlo Benchmark"); logging.info("="*50)
  logging.info("Configuration:"); logging.info(f"  Target Backend: {args.backend.upper()}")
  if args.backend == "gpu":
      if selected_platform_name and selected_device_name:
          logging.info(f"  GPU Platform: {selected_platform_name}"); logging.info(f"  GPU Device: {selected_device_name}")
      else: logging.warning("  GPU Platform/Device: Name capture failed.")
  log_threads_message = f"  CPU Threads requested: {args.threads}"
  if args.backend == "gpu": log_threads_message += " (Note: For GPU, kernel parallelism by OpenCL work-items.)"
  logging.info(log_threads_message)
  logging.info(f"  Total paths: {args.num_paths}"); logging.info(f"  Data size per path: {args.data_size}")
  logging.info(f"  Estimated FLOPs per element per step: {FLOPS_PER_ELEMENT_IN_STEP}"); logging.info("-"*50)
  try:
    duration, total_sum_from_run = run_benchmark(args.num_paths,args.data_size,args.threads,args.backend,cl_context=context,cl_queue=queue)
    logging.info("-"*50); logging.info("Benchmark Execution Summary:")
    logging.info(f"  Backend Used: {args.backend.upper()}")
    if args.backend == "gpu" and selected_platform_name and selected_device_name:
        logging.info(f"  GPU Platform Used: {selected_platform_name}"); logging.info(f"  GPU Device Used: {selected_device_name}")
    logging.info(f"  Core computation time: {duration:.4f} seconds")
    logging.info(f"  Aggregated sum: {total_sum_from_run:.4e}")
    if duration > 0:
        paths_per_second = args.num_paths / duration
        total_fp_ops = args.num_paths * args.data_size * FLOPS_PER_ELEMENT_IN_STEP
        mops = total_fp_ops / duration / 1_000_000; gflops = mops / 1000
        logging.info("  --- Performance Metrics (core computation time) ---")
        logging.info(f"  Paths per second: {paths_per_second:.2f} paths/s")
        logging.info(f"  Estimated MOPS: {mops:.4f} MOPS"); logging.info(f"  Estimated GFLOPs: {gflops:.4f} GFLOPs")
    else: logging.warning("Core computation duration zero/negative. Metrics not calculated.")
    logging.info("-"*50)
    if args.backend == "cpu" and args.threads > 1:
        logging.info("Running single-thread CPU comparison...")
        try:
            s_duration,s_sum = run_benchmark(args.num_paths,args.data_size,1,"cpu")
            logging.info(f"  Single-thread CPU duration: {s_duration:.4f}s, Sum: {s_sum:.4e}")
            if s_duration > 0 and duration > 0:
                speedup = s_duration / duration; efficiency = (speedup / args.threads) * 100
                logging.info(f"  CPU Speedup ({args.threads}t vs 1t): {speedup:.2f}x, Efficiency: {efficiency:.2f}%")
            else: logging.warning("Could not calculate CPU speedup.")
        except Exception as e_single: logging.error(f"Error in single-thread CPU comparison: {e_single}", exc_info=True)
        logging.info("-"*50)
  except MemoryError: logging.critical("Benchmark failed: MemoryError.", exc_info=True)
  except cl.Error as cle: logging.critical(f"OpenCL error in benchmark: {cle.what()} (Code: {cle.code})", exc_info=True)
  except Exception as e: logging.critical(f"Critical error in benchmark: {e}", exc_info=True)
  logging.info("Benchmark script finished.")
