#!/usr/bin/env python3
"""
Basic test of CodeContext AI structure without heavy ML dependencies
"""

import sys
import os
from pathlib import Path

# Add the codecontext_ai module to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_structure():
    """Test basic project structure"""
    print("🔍 Testing CodeContext AI basic structure...")
    
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
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
    
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
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
    
    # Test 3: Test basic Python structure
    try:
        import codecontext_ai
        print("❌ codecontext_ai imports (requires ML dependencies)")
    except ImportError as e:
        print(f"⚠️  codecontext_ai import failed (expected): {e}")
    
    # Test 4: Check if core files have content
    if Path("README.md").exists():
        readme_content = Path("README.md").read_text()
        if "PRIVACY REVOLUTION" in readme_content:
            print("✅ README.md has privacy-focused content")
        else:
            print("❌ README.md missing privacy content")
    
    # Test 5: Check CLI structure
    cli_path = Path("codecontext_ai/cli.py")
    if cli_path.exists():
        print("✅ CLI module exists")
        cli_content = cli_path.read_text()
        if "def main()" in cli_content:
            print("✅ CLI has main function")
        else:
            print("❌ CLI missing main function")
    
    print("\n🎯 Basic structure test complete!")
    return True

def test_configs():
    """Test configuration files"""
    print("\n🔍 Testing configuration files...")
    
    config_files = ["readme.yaml", "api.yaml", "changelog.yaml"]
    
    for config_file in config_files:
        config_path = Path(f"configs/{config_file}")
        if config_path.exists():
            print(f"✅ {config_file} exists")
            try:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                if "model" in config and "training" in config:
                    print(f"✅ {config_file} has required structure")
                else:
                    print(f"❌ {config_file} missing required sections")
            except Exception as e:
                print(f"⚠️  {config_file} YAML parse issue: {e}")
        else:
            print(f"❌ {config_file} missing")

def test_scripts():
    """Test script files"""
    print("\n🔍 Testing script files...")
    
    script_files = [
        "prepare_dataset.py",
        "convert_to_gguf.py", 
        "upload_to_hub.py",
        "benchmark_all.py"
    ]
    
    for script_file in script_files:
        script_path = Path(f"scripts/{script_file}")
        if script_path.exists():
            print(f"✅ {script_file} exists")
            script_content = script_path.read_text()
            if "def main(" in script_content or "if __name__ == \"__main__\":" in script_content:
                print(f"✅ {script_file} has executable structure")
            else:
                print(f"⚠️  {script_file} may not be executable")
        else:
            print(f"❌ {script_file} missing")

if __name__ == "__main__":
    print("🔒 CodeContext AI - Basic Structure Test")
    print("=" * 50)
    
    test_basic_structure()
    test_configs() 
    test_scripts()
    
    print("\n🚀 Test complete! This validates the project structure.")
    print("📝 Note: Full ML functionality requires dependencies installation.")
    print("💡 Run 'make setup-dev' to install all dependencies for training.")