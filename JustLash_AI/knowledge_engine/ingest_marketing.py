#!/usr/bin/env python3
import os
import sys
# Ensure the project root is in sys.path for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from knowledge_engine.diamond_ingestor import ingest_file

def main():
    # Absolute path to the marketing references file in the scratch area
    src_path = "/Users/joseluis/.gemini/antigravity-ide/brain/46d690b1-2ec2-47b7-9839-21d85444143e/scratch/referencias_marketing.txt"
    if os.path.isfile(src_path):
        print(f"Ingesting marketing references from {src_path}")
        ingest_file(src_path)
    else:
        print(f"Reference file not found: {src_path}")

if __name__ == "__main__":
    main()
