#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path

def convert_to_gguf(model_path: str, output_path: str = None, quantization: str = "q4_0"):
    """Convert HuggingFace model to GGUF format for Ollama compatibility"""
    
    if output_path is None:
        output_path = f"{model_path}.gguf"
    
    # Ensure model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Convert to GGUF using llama.cpp tools
    convert_script = "convert-hf-to-gguf.py"
    
    cmd = [
        "python", convert_script,
        model_path,
        "--outfile", output_path,
        "--outtype", quantization
    ]
    
    print(f"Converting {model_path} to GGUF format...")
    print(f"Output: {output_path}")
    print(f"Quantization: {quantization}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Successfully converted to {output_path}")
        
        # Validate output file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024**3)  # GB
            print(f"📦 Model size: {file_size:.2f} GB")
        else:
            print("❌ Output file not created")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        raise
    except FileNotFoundError:
        print("❌ llama.cpp conversion tools not found")
        print("Install from: https://github.com/ggerganov/llama.cpp")
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert model to GGUF format")
    parser.add_argument("--model", required=True, help="Path to HuggingFace model")
    parser.add_argument("--output", help="Output GGUF file path")
    parser.add_argument("--quantization", default="q4_0", 
                       choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"],
                       help="Quantization type")
    
    args = parser.parse_args()
    
    convert_to_gguf(args.model, args.output, args.quantization)

if __name__ == "__main__":
    main()