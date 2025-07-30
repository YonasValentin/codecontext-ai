#!/usr/bin/env python3
"""
Check model file sizes and ensure they meet repository standards
"""

import sys
import argparse
from pathlib import Path


def check_model_size(file_path: str, max_size_gb: float = 10.0) -> bool:
    """
    Check if model file size is within acceptable limits
    
    Args:
        file_path: Path to model file
        max_size_gb: Maximum allowed size in gigabytes
        
    Returns:
        True if size is acceptable, False otherwise
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"ERROR: Model file not found: {file_path}")
        return False
    
    # Get file size in GB
    size_bytes = path.stat().st_size
    size_gb = size_bytes / (1024 ** 3)
    
    print(f"Model: {path.name}")
    print(f"Size: {size_gb:.2f} GB ({size_bytes:,} bytes)")
    
    if size_gb > max_size_gb:
        print(f"ERROR: Model size ({size_gb:.2f} GB) exceeds limit ({max_size_gb} GB)")
        print("Consider using quantization to reduce model size:")
        print("  - q4_0: ~4-bit quantization (best compression)")
        print("  - q8_0: ~8-bit quantization (good balance)")
        print("  - fp16: Half precision (minimal compression)")
        return False
    
    print(f"✓ Model size is within acceptable limits")
    return True


def main():
    parser = argparse.ArgumentParser(description="Check model file sizes")
    parser.add_argument("files", nargs="*", help="Model files to check")
    parser.add_argument("--max-size", type=float, default=10.0, 
                       help="Maximum size in GB (default: 10.0)")
    parser.add_argument("--all", action="store_true",
                       help="Check all .gguf files in current directory")
    
    args = parser.parse_args()
    
    files_to_check = []
    
    if args.all:
        files_to_check = list(Path(".").rglob("*.gguf"))
    elif args.files:
        files_to_check = [Path(f) for f in args.files]
    else:
        print("No files specified. Use --all or provide file paths.")
        return 1
    
    if not files_to_check:
        print("No model files found.")
        return 0
    
    all_valid = True
    
    for file_path in files_to_check:
        print("\n" + "="*50)
        if not check_model_size(str(file_path), args.max_size):
            all_valid = False
    
    print("\n" + "="*50)
    if all_valid:
        print("✓ All model files are within size limits")
        return 0
    else:
        print("✗ Some model files exceed size limits")
        return 1


if __name__ == "__main__":
    sys.exit(main())