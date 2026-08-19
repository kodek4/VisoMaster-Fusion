"""Export the official AlphaFace checkpoint to VisoMaster's ONNX format.

Usage:
    python app/tools/export_alphaface_onnx.py path/to/alphaface_demo.pt

The official checkpoint can be obtained from the AlphaFace repository:
https://github.com/andrewyu90/Alphaface_Official
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import onnx
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.processors.alphaface import AlphaFaceSwapper  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "model_assets" / "alphaface" / "alphaface_swapper.onnx"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        while chunk := model_file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = (
        checkpoint.get("swapper", checkpoint) if isinstance(checkpoint, dict) else None
    )
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain an AlphaFace swapper state dict")
    return state


def export(checkpoint_path: Path, output_path: Path) -> None:
    model = AlphaFaceSwapper().eval()
    model.load_state_dict(_checkpoint_state(checkpoint_path), strict=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".exporting.onnx")
    target = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    identity = torch.zeros((1, 512), dtype=torch.float32)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (target, identity),
            temporary_path,
            input_names=["target", "source_embedding"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    graph = onnx.load(temporary_path, load_external_data=False)
    onnx.checker.check_model(graph)
    del graph
    temporary_path.replace(output_path)

    digest = _sha256(output_path)
    print(f"Exported {output_path}")
    print(f"SHA256: {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Official alphaface_demo.pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.checkpoint.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
