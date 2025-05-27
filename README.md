feat: Add disk benchmark script for Linux

This script, `disk_benchmark.py`, measures disk write performance.

Key Features:
- Detects filesystem type and native block size for a target directory.
- Performs sequential and random write benchmarks using various block sizes to determine optimal performance.
- Calculates write speeds in MB/s and IOPS.
- Reports the best-performing block sizes for both write patterns.
- Allows customization of target directory, test file size, test duration, and specific block sizes via command-line arguments.
- Includes comprehensive error handling for issues like invalid paths, permissions, and missing system commands.
- Provides detailed usage instructions and examples through the --help option.

The script helps you understand your disk's write capabilities and choose appropriate block sizes for your applications on Linux (Red Hat focused) systems.
