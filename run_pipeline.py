import argparse
import shutil
from pathlib import Path

from clin2pro.config import load_config
from clin2pro.stage1 import train_stage1
from clin2pro.stage2 import train_stage2
from clin2pro.stage3 import train_stage3
from clin2pro.utils import resolve_device, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Three-stage longitudinal delta-proteomics pipeline."
    )
    parser.add_argument("--config", required=True, help="YAML configuration file.")
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Stages to train. Missing earlier stages are loaded from output_dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config["_config_path"], output / "resolved_config.yaml")
    logger = setup_logging(str(output))
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    logger.info("Device: %s", device)
    stage1_checkpoint = output / "stage1" / "best.pt"
    stage2_checkpoint = output / "stage2" / "best.pt"
    if 1 in args.stages:
        stage1_checkpoint = train_stage1(config, device, logger)
    if not stage1_checkpoint.exists():
        raise FileNotFoundError("Stage 1 checkpoint not found: %s" % stage1_checkpoint)
    if 2 in args.stages:
        stage2_checkpoint = train_stage2(
            config, device, logger, stage1_checkpoint
        )
    if not stage2_checkpoint.exists() and 3 in args.stages:
        raise FileNotFoundError("Stage 2 checkpoint not found: %s" % stage2_checkpoint)
    if 3 in args.stages:
        results = train_stage3(
            config, device, logger, stage1_checkpoint, stage2_checkpoint
        )
        logger.info("Finished. OOF results: %s", results)


if __name__ == "__main__":
    main()
