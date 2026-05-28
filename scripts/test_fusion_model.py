"""
Thin entry-point kept for backward compatibility.
The training infrastructure has moved to training/.

Prefer running:
    python -m training.train [args]
"""
import argparse
from training.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train FusionModel (RNN + CAD encoder) on trajectory data."
    )
    parser.add_argument("--output-dir", type=str, default="outputs/L_shape_fusion_test")
    parser.add_argument("--dataset-path", type=str, default="datasets/L_shape")
    parser.add_argument("--headless", action="store_false")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    train(
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        headless=args.headless,
        eval_only=args.eval_only,
    )
