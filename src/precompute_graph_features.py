import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import ImageDataset, MORPHOLOGY_COLUMNS
from models import DualViewModel


def main(args):
    root = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(root / "metadata.csv")
    mapping = json.loads((root / "label_mapping_coarse.json").read_text())
    class_names = [mapping["coarse_index_to_label"][str(i)] for i in range(len(mapping["coarse_index_to_label"]))]

    model = DualViewModel(len(class_names)).to(device)
    model.load_state_dict(torch.load(args.image_checkpoint, map_location=device)["model"])
    model.eval()

    embeddings = []
    for split in ("train", "val", "test"):
        loader = DataLoader(ImageDataset(root, split), args.batch_size, shuffle=False, num_workers=args.workers)
        with torch.no_grad():
            for batch in loader:
                embeddings.append(model.encode(batch["masked"].to(device)).cpu())

    ordered = pd.concat([metadata[metadata.split == split] for split in ("train", "val", "test")])
    cache = {
        "embeddings": torch.cat(embeddings),
        "coords": ordered[["x_hires", "y_hires"]].to_numpy(np.float32),
        "morphology": ordered[MORPHOLOGY_COLUMNS].to_numpy(np.float32),
        "labels": ordered.coarse_label_index.to_numpy(np.int64),
        "splits": ordered.split.astype(str).to_numpy(),
        "sample_ids": ordered.sample_id.astype(str).to_numpy(),
        "cell_ids": ordered.cell_id.astype(str).to_numpy(),
        "class_names": class_names,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--output", default="checkpoints/graph_features.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args())
