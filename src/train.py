import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from data import FusionDataset, GraphDataset, ImageDataset
from models import DualViewModel, HierarchicalFusion, NeighborGraphModel


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_names(data_dir):
    mapping = json.loads((Path(data_dir) / "label_mapping_coarse.json").read_text())
    labels = mapping["coarse_index_to_label"]
    return [labels[str(i)] for i in range(len(labels))]


def class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def loaders(args):
    if args.stage == "image":
        datasets = [ImageDataset(args.data_dir, split, split == "train") for split in ("train", "val", "test")]
    elif args.stage == "graph":
        datasets = [GraphDataset(args.graph_cache, split, args.scales) for split in ("train", "val", "test")]
    else:
        datasets = [FusionDataset(args.data_dir, args.graph_cache, split, args.scales, False) for split in ("train", "val", "test")]
    return [
        DataLoader(dataset, args.batch_size, shuffle=i == 0, num_workers=args.workers, pin_memory=True)
        for i, dataset in enumerate(datasets)
    ], datasets


def move_graphs(graphs, device):
    return [graph.to(device, non_blocking=True) for graph in graphs]


def forward(model, batch, stage, device):
    if stage == "image":
        labels = batch["label"].to(device)
        outputs = model(batch["raw"].to(device), batch["masked"].to(device))
    elif stage == "graph":
        labels = batch["label"].to(device)
        outputs = model(move_graphs(batch["graphs"], device))
    else:
        images, graph = batch
        labels = images["label"].to(device)
        outputs = model(
            images["raw"].to(device),
            images["masked"].to(device),
            move_graphs(graph["graphs"], device),
        )
    return outputs, labels


def run_epoch(model, loader, stage, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    labels_all, predictions_all, probabilities_all = [], [], []
    total_loss = 0

    for batch in loader:
        with torch.set_grad_enabled(optimizer is not None):
            outputs, labels = forward(model, batch, stage, device)
            loss = criterion(outputs["logits"], labels)
            if stage == "graph":
                composition = batch["composition"].to(device)
                loss += 0.1 * -(composition * outputs["composition"].log()).sum(1).mean()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probabilities = outputs["logits"].softmax(1).detach().cpu()
        total_loss += loss.item() * len(labels)
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(probabilities.argmax(1).tolist())
        probabilities_all.append(probabilities)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels_all, predictions_all),
        "balanced_accuracy": balanced_accuracy_score(labels_all, predictions_all),
        "macro_f1": f1_score(labels_all, predictions_all, average="macro"),
    }
    return metrics, labels_all, torch.cat(probabilities_all)


def build_model(args, datasets, num_classes, device):
    if args.stage == "image":
        return DualViewModel(num_classes).to(device)

    node_dim = datasets[0][0]["graphs"][0].shape[1]
    graph = NeighborGraphModel(node_dim, num_classes, len(args.scales)).to(device)
    if args.stage == "graph":
        return graph

    image = DualViewModel(num_classes).to(device)
    image.load_state_dict(torch.load(args.image_checkpoint, map_location=device)["model"])
    graph.load_state_dict(torch.load(args.graph_checkpoint, map_location=device)["model"])
    return HierarchicalFusion(image, graph).to(device)


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = class_names(args.data_dir)
    (train_loader, val_loader, test_loader), datasets = loaders(args)
    model = build_model(args, datasets, len(names), device)

    train_labels = datasets[0].images.frame.coarse_label_index.to_numpy() if args.stage == "fusion" else (
        datasets[0].frame.coarse_label_index.to_numpy() if args.stage == "image" else datasets[0].labels[datasets[0].indices]
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_labels, len(names), device))

    if args.stage == "image":
        encoder = list(model.encoder.parameters())
        head = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]
        optimizer = torch.optim.AdamW([{"params": encoder, "lr": 1e-5}, {"params": head, "lr": args.lr}], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_f1 = -1
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, args.stage, criterion, device, optimizer)
        val_metrics, _, _ = run_epoch(model, val_loader, args.stage, criterion, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"epoch={epoch:02d} train_f1={train_metrics['macro_f1']:.4f} val_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "class_names": names, "scales": args.scales}, output / "best_model.pth")

    model.load_state_dict(torch.load(output / "best_model.pth", map_location=device)["model"])
    test_metrics, labels, probabilities = run_epoch(model, test_loader, args.stage, criterion, device)
    (output / "metrics.json").write_text(json.dumps({"best_val_macro_f1": best_f1, "test": test_metrics}, indent=2))
    pd.DataFrame({
        "label": labels,
        "prediction": probabilities.argmax(1).numpy(),
        **{f"prob_{name}": probabilities[:, i].numpy() for i, name in enumerate(names)},
    }).to_csv(output / "test_predictions.csv", index=False)
    (output / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the image, graph, or fusion stage.")
    parser.add_argument("stage", choices=["image", "graph", "fusion"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--graph-cache")
    parser.add_argument("--image-checkpoint")
    parser.add_argument("--graph-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scales", type=lambda value: tuple(map(int, value.split(","))), default=(8, 24, 64))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
