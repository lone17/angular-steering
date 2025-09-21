import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm

REQUIRED_KEYS = ["prompt", "response", "generated_token_ids", "token_activations"]
OUTPUT_JSON_KEYS = ["prompt", "response", "generated_token_ids"]


def load_sequences_activations(file_path: str) -> dict:
    return torch.load(file_path)

def get_sample_idx_from_file_path(file_path: str) -> int:
    file_name = Path(file_path).name
    return int(file_name.split("_")[-1].split(".")[0])


def split_sequences_activations(act_path: str, output_dir: str, remove_original: bool = False) -> tuple:
    data = load_sequences_activations(act_path)
    sample_idx = get_sample_idx_from_file_path(act_path)
    
    data_keys = list(data.keys())
    if not all([key in data_keys for key in REQUIRED_KEYS]):
        raise ValueError(f"Sequences activations must contain the following keys: {REQUIRED_KEYS}, got {data_keys}")
    
    generated_token_acts = data.pop("token_activations")

    generated_pt_path = output_dir / f"generation_activations_{sample_idx}.pt"
    torch.save(generated_token_acts, generated_pt_path)

    json_path = output_dir / f"response_{sample_idx}.json"
    with open(json_path, "w") as f:
        data["generated_token_ids"] = data["generated_token_ids"].tolist()
        json.dump(data, f, indent=4)

    if remove_original:
        os.remove(act_path)
        print(f"Removed {act_path}")

    return generated_pt_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(description="Split sequences activations")
    parser.add_argument("--acts_dir", type=str, help="Path to the sequences activations *.pt files")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory after splitting")
    parser.add_argument("--remove_original", action="store_true", help="Remove the original sequences activations *.pt files after splitting")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    acts_dir = Path(args.acts_dir)
    act_paths = list(acts_dir.glob("*.pt"))
    for act_path in tqdm(act_paths, total=len(act_paths), desc="Extracting sequences activations"):
        generated_pt_path, json_path = split_sequences_activations(act_path, output_dir, args.remove_original)
        print(f"Split {act_path} into {generated_pt_path} and {json_path}")
        print("--------------------------------")