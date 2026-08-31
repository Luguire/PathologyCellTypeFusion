from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from torchvision import transforms


MORPHOLOGY_COLUMNS = [
    "mask_area",
    "attention_mask_area",
    "mask_bbox_width",
    "mask_bbox_height",
    "fullres_polygon_area",
]


def image_transform(train=False):
    steps = [transforms.Resize((224, 224))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.15, 0.15, 0.10, 0.02),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(steps)


class ImageDataset(Dataset):
    def __init__(self, data_dir, split, train=False):
        self.root = Path(data_dir)
        self.frame = pd.read_csv(self.root / "metadata.csv")
        self.frame = self.frame[self.frame.split == split].reset_index(drop=True)
        self.transform = image_transform(train)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        raw = Image.open(self.root / row.raw_patch_path).convert("RGB")
        masked = Image.open(self.root / row.masked_target_patch_path).convert("RGB")
        return {
            "raw": self.transform(raw),
            "masked": self.transform(masked),
            "label": torch.tensor(row.coarse_label_index, dtype=torch.long),
            "sample_id": str(row.sample_id),
            "cell_id": str(row.cell_id),
        }


class GraphDataset(Dataset):
    def __init__(self, cache_path, split, scales=(8, 24, 64)):
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
        self.coords = np.asarray(cache["coords"], dtype=np.float32)
        self.morphology = np.asarray(cache["morphology"], dtype=np.float32)
        self.labels = np.asarray(cache["labels"], dtype=np.int64)
        self.splits = np.asarray(cache["splits"])
        self.sample_ids = np.asarray(cache["sample_ids"])
        self.cell_ids = np.asarray(cache["cell_ids"])
        self.class_names = cache["class_names"]
        self.scales = tuple(scales)
        self.indices = np.flatnonzero(self.splits == split)

        train = self.splits == "train"
        mean = self.morphology[train].mean(0)
        std = self.morphology[train].std(0)
        self.morphology = (self.morphology - mean) / np.maximum(std, 1e-6)
        self.coord_scale = self.coords[train].std(0).mean()

        pool = np.flatnonzero(self.splits == split)
        knn = NearestNeighbors(n_neighbors=max(self.scales) + 1).fit(self.coords[pool])
        local = knn.kneighbors(self.coords[self.indices], return_distance=False)
        self.neighbors = np.stack([
            pool[row[row != target]][: max(self.scales)]
            for row, target in zip(local, self.indices)
        ])

    def __len__(self):
        return len(self.indices)

    def _graph(self, target, neighbors, k):
        node_ids = np.concatenate([[target], neighbors[:k]])
        relative = (self.coords[node_ids] - self.coords[target]) / self.coord_scale
        target_flag = np.zeros((k + 1, 1), dtype=np.float32)
        target_flag[0] = 1
        features = np.concatenate([
            self.embeddings[node_ids], self.morphology[node_ids], relative, target_flag
        ], axis=1)
        return torch.tensor(features, dtype=torch.float32)

    def __getitem__(self, index):
        target = self.indices[index]
        neighbors = self.neighbors[index]
        composition = np.bincount(self.labels[neighbors], minlength=len(self.class_names))
        composition = composition / composition.sum()
        return {
            "graphs": [self._graph(target, neighbors, k) for k in self.scales],
            "label": torch.tensor(self.labels[target], dtype=torch.long),
            "composition": torch.tensor(composition, dtype=torch.float32),
            "sample_id": str(self.sample_ids[target]),
            "cell_id": str(self.cell_ids[target]),
        }


class FusionDataset(Dataset):
    def __init__(self, data_dir, cache_path, split, scales=(8, 24, 64), train=False):
        self.images = ImageDataset(data_dir, split, train)
        self.graphs = GraphDataset(cache_path, split, scales)
        graph_index = {
            (self.graphs.sample_ids[i], self.graphs.cell_ids[i]): j
            for j, i in enumerate(self.graphs.indices)
        }
        self.graph_rows = [
            graph_index[(str(row.sample_id), str(row.cell_id))]
            for _, row in self.images.frame.iterrows()
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.graphs[self.graph_rows[index]]

