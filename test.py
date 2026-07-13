"""
Usage:
    python test.py --model_path test_model_r2_split_aa_range3.pt \
                    --test_data new_test_value.pkl
"""

import argparse
import pickle

import torch
from torch_geometric.loader import DataLoader

from data_split import ProteinPHValueGraphDataset
from train_model_value_split_range import evaluate_model


def get_test_dataloader(test_data_path, radius, batch_size, num_workers):
    print(f"Loading test set from {test_data_path}...")
    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)

    test_dataset = ProteinPHValueGraphDataset(test_data, radius=radius)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        prefetch_factor=2,
    )
    return test_dataloader


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DeepPH model on a test set."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the trained model checkpoint (.pt)"
    )
    parser.add_argument(
        "--test_data", type=str, required=True,
        help="Path to the test set (.pkl), e.g. new_test_value.pkl or "
             "new_test_value_remove_phenv.pkl"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--radius", type=float, default=15)
    parser.add_argument(
        "--device", type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    test_dataloader = get_test_dataloader(
        args.test_data, args.radius, args.batch_size, args.num_workers
    )

    print(f"Loading model from {args.model_path}...")
    model = torch.load(args.model_path, map_location=args.device, weights_only=False)
    model = model.to(args.device)
    model.eval()

    print("Running evaluation...")
    evaluate_model(model, test_dataloader)


if __name__ == "__main__":
    main()