import argparse
import sys
from pathlib import Path

# Support direct execution from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clin2pro.config import load_config
from clin2pro.stage1 import (
    export_stage1_oof_predictions,
    export_stage1_predictions,
)
from clin2pro.utils import resolve_device, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export all Stage 1 cross-sectional predictions and held-out "
            "validation metrics from an existing checkpoint."
        )
    )
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional Stage 1 checkpoint. Defaults to outputs/full/stage1/best.pt.",
    )
    parser.add_argument(
        "--oof-folds",
        type=int,
        default=5,
        help="Number of folds for unbiased all-participant OOF predictions.",
    )
    parser.add_argument(
        "--skip-oof",
        action="store_true",
        help="Only export fitted/all and held-out predictions from the checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["device"])
    logger = setup_logging(config["output_dir"])
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    output = export_stage1_predictions(
        config,
        device,
        logger,
        checkpoint_path=checkpoint,
    )
    logger.info("Cross-sectional prediction export completed: %s", output)
    if not args.skip_oof:
        oof_output = export_stage1_oof_predictions(
            config,
            device,
            logger,
            folds=args.oof_folds,
        )
        logger.info("Cross-sectional OOF export completed: %s", oof_output)


if __name__ == "__main__":
    main()
