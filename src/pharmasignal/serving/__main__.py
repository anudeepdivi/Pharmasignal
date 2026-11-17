"""Run local inference: python -m pharmasignal.serving --model-dir PATH."""

import argparse
import os
from pathlib import Path

import uvicorn

from pharmasignal.serving.app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("PHARMASIGNAL_MODEL_DIR", "artifacts/models/m4")),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # Access logs contain URLs/query strings, so disable them as well as avoiding
    # body logging. Single worker, no reload and no remote model fetching.
    uvicorn.run(create_app(args.model_dir), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
