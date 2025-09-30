#!/usr/bin/env python3
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from step1_get_activations import SequenceActivations
from tqdm import tqdm


def get_sample_index(filename: str) -> int:
    """Extract sample index from filename like sequences_activations_{index}.pt"""
    match = re.search(r"sequences_activations_(\d+)\.pt", filename)
    if match:
        return int(match.group(1))
    return -1


def process_folder(folder_path: Path) -> None:
    """Process all activation files in a folder and save lengths.txt"""
    print(f"Processing folder: {folder_path}")

    # Check if folder exists
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Folder {folder_path} does not exist or is not a directory")
        return

    # Get all activation files
    activation_files = list(folder_path.glob("sequences_activations_*.pt"))
    if not activation_files:
        print(f"No activation files found in {folder_path}")
        return

    # Collect lengths and indices
    length_index_pairs: List[Tuple[int, int]] = []

    for file_path in tqdm(activation_files, desc="Processing files"):
        try:
            # Load the file
            data = torch.load(file_path)

            # Extract sample index from filename
            sample_index = get_sample_index(file_path.name)

            # If loaded as dict (from asdict), create SequenceActivations
            if isinstance(data, dict):
                seq_activations = SequenceActivations(**data)
                # Get length of generated tokens
                token_length = len(seq_activations.generated_token_ids[0])
            else:
                # Assume it's already a SequenceActivations object
                token_length = len(data.generated_token_ids[0])

            # Add to list
            length_index_pairs.append((token_length, sample_index))

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Sort by length
    length_index_pairs.sort()

    # Save to file
    output_file = folder_path / "lengths.txt"
    with open(output_file, "w") as f:
        # Write each length and index on a separate line without tuple formatting
        for length, index in length_index_pairs:
            f.write(f"{length}, {index}\n")

    print(f"Saved lengths to {output_file}")


def main():
    # Base directory
    base_dir = Path(
        "/home/ian/repos/thinking_control/thinking_models/outputs/thinking-model-activations"
    )

    models = [
        # "Qwen/Qwen3-4B",
        # "Qwen/Qwen3-8B",
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-LLama-8B",
    ]

    # Iterate through model directories
    # for model_dir in base_dir.iterdir():
    for model_id in models:
        model_name = model_id.split("/")[-1]
        model_dir = base_dir / model_name
        if not model_dir.is_dir():
            continue

        print(f"Processing model: {model_dir.name}")

        # Iterate through activation type directories
        for activation_dir in model_dir.iterdir():
            if not activation_dir.is_dir():
                continue

            process_folder(activation_dir)


if __name__ == "__main__":
    main()
