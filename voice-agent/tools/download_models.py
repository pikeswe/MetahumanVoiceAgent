"""Helper to download Piper voice models."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US-amy-medium.onnx?download=true"
DEFAULT_OUT = Path("models/piper/en_US-amy-medium.onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Piper voice model")
    parser.add_argument("--url", default=DEFAULT_URL, help="Voice model URL")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output path")
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.url} -> {out_path}")
    urlretrieve(args.url, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
