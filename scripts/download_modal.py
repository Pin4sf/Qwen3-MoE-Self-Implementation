# scripts/download_model.py
import argparse
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import download_and_load_weights

def main():
    parser = argparse.ArgumentParser(description="Download a Qwen3 model from Hugging Face.")
    parser.add_argument(
        "--repo_id", 
        type=str, 
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct", 
        help="The repository ID of the model to download from Hugging Face."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="models", 
        help="The local directory to save the model files."
    )
    args = parser.parse_args()

    try:
        download_and_load_weights(args.repo_id, args.save_dir)
        print(f"\nSuccessfully downloaded model '{args.repo_id}' to '{args.save_dir}'.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()