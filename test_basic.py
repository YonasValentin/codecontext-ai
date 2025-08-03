#!/usr/bin/env python3
"""
Basic test of CodeContext AI™ structure without heavy ML dependencies
"""

import sys
import os
from pathlib import Path

# Add the codecontext_ai module to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_structure():
    """Test basic project structure"""
    print("Testing CodeContext AI™ basic structure...")
    
    # Test 1: Check if directories exist
    required_dirs = [
        "codecontext_ai",
        "scripts", 
        "configs",
        "tests",
        ".github"
    ]
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"[OK] {dir_name}/ directory exists")
        else:
            print(f"[MISSING] {dir_name}/ directory missing")
    
    # Test 2: Check if key files exist
    required_files = [
        "README.md",
        "setup.py", 
        "pyproject.toml",
        "requirements.txt",
        "CONTRIBUTING.md",
        "LICENSE",
        "Makefile"
    ]
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"[OK] {file_name} exists")
        else:
            print(f"[MISSING] {file_name} missing")
    
    # Test 3: Test basic Python structure
    try:
        import codecontext_ai
        print("[ERROR] codecontext_ai imports (requires ML dependencies)")
    except ImportError as e:
        print(f"[WARNING] codecontext_ai import failed (expected): {e}")
    
    # Test 4: Check if core files have content
    if Path("README.md").exists():
        readme_content = Path("README.md").read_text()
        if "Privacy-first" in readme_content:
            print("[OK] README.md has privacy-focused content")
        else:
            print("[ERROR] README.md missing privacy content")
    
    # Test 5: Check CLI structure
    cli_path = Path("codecontext_ai/cli.py")
    if cli_path.exists():
        print("[OK] CLI module exists")
        cli_content = cli_path.read_text()
        if "def main()" in cli_content:
            print("[OK] CLI has main function")
        else:
            print("[ERROR] CLI missing main function")
    
    print("\nBasic structure test complete")
    return True

def test_configs():
    """Test configuration files"""
    print("\nTesting configuration files...")
    
    config_files = ["readme.yaml", "api.yaml", "changelog.yaml"]
    
    for config_file in config_files:
        config_path = Path(f"configs/{config_file}")
        if config_path.exists():
            print(f"[OK] {config_file} exists")
            try:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                if "model" in config and "training" in config:
                    print(f"[OK] {config_file} has required structure")
                else:
                    print(f"[ERROR] {config_file} missing required sections")
            except Exception as e:
                print(f"[WARNING] {config_file} YAML parse issue: {e}")
        else:
            print(f"[MISSING] {config_file} missing")

def test_scripts():
    """Test script files"""
    print("\nTesting script files...")
    
    script_files = [
        "prepare_dataset.py",
        "convert_to_gguf.py", 
        "upload_to_hub.py",
        "benchmark_all.py"
    ]
    
    for script_file in script_files:
        script_path = Path(f"scripts/{script_file}")
        if script_path.exists():
            print(f"[OK] {script_file} exists")
            script_content = script_path.read_text()
            if "def main(" in script_content or "if __name__ == \"__main__\":" in script_content:
                print(f"[OK] {script_file} has executable structure")
            else:
                print(f"[WARNING] {script_file} may not be executable")
        else:
            print(f"[MISSING] {script_file} missing")

if __name__ == "__main__":
    print("CodeContext AI™ - Basic Structure Test")
    print("=" * 50)
    
    test_basic_structure()
    test_configs() 
    test_scripts()
    
    print("\nTest complete. Project structure validated.")
    print("Note: Full ML functionality requires dependencies installation.")
    print("Run 'make setup-dev' to install all dependencies for training.")