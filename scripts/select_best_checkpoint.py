import argparse
import glob
import os
import re
import shutil


def find_best_checkpoint(checkpoints_dir: str) -> str:
    pattern = os.path.join(checkpoints_dir, "*.ckpt")
    best_path = None
    best_loss = float("inf")

    for path in glob.glob(pattern):
        match = re.search(r"val_loss=([0-9]+(?:\.[0-9]+)?)", os.path.basename(path))
        if not match:
            continue
        loss = float(match.group(1))
        if loss < best_loss:
            best_loss = loss
            best_path = path

    if not best_path:
        raise FileNotFoundError(
            f"No checkpoint with val_loss found in {checkpoints_dir}"
        )

    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best checkpoint by val_loss")
    parser.add_argument(
        "--checkpoints-dir",
        default="./checkpoints",
        help="Directory with .ckpt files",
    )
    parser.add_argument(
        "--output",
        default="./artifacts/best.ckpt",
        help="Path to copy the best checkpoint",
    )
    args = parser.parse_args()

    best_path = find_best_checkpoint(args.checkpoints_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    shutil.copy2(best_path, args.output)
    print(f"Best checkpoint: {best_path}")
    print(f"Copied to: {args.output}")


if __name__ == "__main__":
    main()
