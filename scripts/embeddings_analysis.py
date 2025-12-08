"""Embedding analysis utilities: t-SNE projection and Deep kNN retrieval.

Example usage:
    python scripts/embeddings_analysis.py --tsne --run-deepknn \
        --gallery-dir dataset/classification_data/train_data \
        --query-dir dataset/classification_data/val_data \
        --model-path outputs/best_metric_model.pth --mode metric
"""

from __future__ import annotations

import os
# fix duplicate OpenMP runtime on Windows (libiomp5md.dll)
# set before importing libraries that load OpenMP (e.g., torch, cv2)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

# Ensure src/ is on the path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import DEVICE, EMBEDDING_DIM, OUTPUT_DIR  # type: ignore  # noqa: E402
from data_loader import get_transforms  # type: ignore  # noqa: E402
from deep_knn import evaluate_topk, fit_knn, knn_search  # type: ignore  # noqa: E402
from models import FaceEmbeddingCNN  # type: ignore  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ImagePathDataset(Dataset):
    """Minimal dataset wrapper around image paths and labels."""

    def __init__(self, paths: Sequence[Path], labels: Sequence[int], transform):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding analysis utilities")
    parser.add_argument("--gallery-dir", type=Path, default=ROOT_DIR / "dataset" / "classification_data" / "train_data")
    parser.add_argument("--query-dir", type=Path, default=None, help="Optional directory for query images (defaults to gallery)")
    parser.add_argument("--model-path", type=Path, default=ROOT_DIR / "outputs" / "best_metric_model.pth")
    parser.add_argument("--mode", choices=["metric", "embedding", "classification"], default="metric")
    parser.add_argument("--output-prefix", type=Path, default=None, help="Override output prefix (directory or file stem)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Limit total samples (set to -1 to disable)",
    )
    parser.add_argument("--limit-per-class", type=int, default=None, help="Optional limit per identity")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--reduction", choices=["tsne", "none"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--pca-components", type=int, default=None, help="Optional PCA dimensionality before t-SNE")
    parser.add_argument("--tsne-learning-rate", type=str, default="auto")

    parser.add_argument("--run-deepknn", action="store_true", help="Enable Deep kNN evaluation")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--knn-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--knn-visualize-count", type=int, default=3)

    return parser.parse_args()


def build_output_path(prefix: Path, suffix: str) -> Path:
    """Return a file path by appending a suffix to the prefix stem."""

    return prefix.parent / f"{prefix.name}{suffix}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gather_image_paths(directory: Path, limit_per_class: Optional[int] = None) -> Tuple[List[Path], List[int], List[str]]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    image_paths: List[Path] = []
    labels: List[int] = []
    class_names: List[str] = []

    for class_idx, class_dir in enumerate(sorted(p for p in directory.iterdir() if p.is_dir())):
        images = [img for img in sorted(class_dir.iterdir()) if img.suffix.lower() in VALID_EXTENSIONS]
        if not images:
            continue
        if limit_per_class is not None:
            images = images[:limit_per_class]
        image_paths.extend(images)
        labels.extend([class_idx] * len(images))
        class_names.append(class_dir.name)

    if not image_paths:
        raise RuntimeError(f"No images discovered under {directory}")

    return image_paths, labels, class_names


def sample_subset(paths: List[Path], labels: List[int], sample_size: Optional[int], seed: int) -> Tuple[List[Path], List[int]]:
    if sample_size is None or sample_size <= 0 or sample_size >= len(paths):
        return paths, labels
    rng = random.Random(seed)
    indices = rng.sample(range(len(paths)), sample_size)
    indices.sort()
    sampled_paths = [paths[i] for i in indices]
    sampled_labels = [labels[i] for i in indices]
    return sampled_paths, sampled_labels


def load_model(model_path: Path, num_classes: int) -> FaceEmbeddingCNN:
    """Load embedding model with the correct classification head dimension."""

    model = FaceEmbeddingCNN(embedding_dim=EMBEDDING_DIM, num_classes=num_classes or 4000).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_embeddings(
    model: FaceEmbeddingCNN,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    mode: str,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stream embeddings with minimal copies and worker reuse."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    embedding_store: Optional[np.ndarray] = None
    label_store: Optional[np.ndarray] = None
    offset = 0
    amp_enabled = use_amp and torch.cuda.is_available()

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=amp_enabled):
        for images, batch_labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images, mode=mode)
            if mode == "classification":
                outputs = torch.nn.functional.softmax(outputs, dim=1)

            outputs_np = outputs.detach().cpu().numpy()
            batch_labels_np = batch_labels.numpy()
            batch_size_actual = outputs_np.shape[0]

            if embedding_store is None:
                embedding_store = np.empty((len(dataset), outputs_np.shape[1]), dtype=np.float32)
                label_store = np.empty(len(dataset), dtype=np.int64)

            embedding_store[offset : offset + batch_size_actual] = outputs_np
            label_store[offset : offset + batch_size_actual] = batch_labels_np
            offset += batch_size_actual

    assert embedding_store is not None and label_store is not None
    return embedding_store, label_store


def maybe_run_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    output_prefix: Path,
    perplexity: float,
    learning_rate: str,
    pca_components: Optional[int],
) -> None:
    if embeddings.shape[0] < 10:
        print("Skipping t-SNE: not enough samples (need >= 10)")
        return

    data = embeddings
    if pca_components is not None and 0 < pca_components < embeddings.shape[1]:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=pca_components, random_state=42)
        data = pca.fit_transform(embeddings)
        print(f"Applied PCA to {pca_components} components before t-SNE")

    print("Running t-SNE projection... (this may take a while)")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        metric="euclidean",
        init="pca",
        random_state=42,
        verbose=1,
    )
    coords = tsne.fit_transform(data)
    save_tsne_results(coords, labels, class_names, output_prefix)


def save_tsne_results(
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    output_prefix: Path,
) -> None:
    csv_path = build_output_path(output_prefix, "_tsne.csv")
    png_path = build_output_path(output_prefix, "_tsne.png")

    df = pd.DataFrame(
        {
            "x": coords[:, 0].astype(float),
            "y": coords[:, 1].astype(float),
            "label_id": labels.astype(int),
            "label_name": [class_names[i] for i in labels],
        }
    )
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 8))
    num_classes = len(class_names)
    cmap = _choose_colormap(num_classes)
    scatter = plt.scatter(df["x"], df["y"], c=df["label_id"], cmap=cmap, s=10, alpha=0.85)

    if num_classes <= 20:
        unique_ids = sorted(np.unique(labels))
        handles = [
            Patch(color=cmap(class_id % cmap.N), label=class_names[class_id])
            for class_id in unique_ids
        ]
        plt.legend(handles=handles, loc="best", fontsize=8, title="Identity")
    else:
        plt.title("t-SNE projection (legend omitted due to many classes)")

    if num_classes <= 20:
        plt.title("t-SNE projection")

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Saved t-SNE scatter: {png_path}")
    print(f"Saved t-SNE coordinates: {csv_path}")


def run_deep_knn(
    embeddings_gallery: np.ndarray,
    labels_gallery: np.ndarray,
    paths_gallery: Sequence[Path],
    class_names: Sequence[str],
    embeddings_query: np.ndarray,
    labels_query: np.ndarray,
    paths_query: Sequence[Path],
    output_prefix: Path,
    k: int,
    metric: str,
    visualize_count: int,
    exclude_self: bool,
) -> None:
    knn_model = fit_knn(embeddings_gallery, metric=metric, k=max(k, 5))
    distances, neighbour_labels, neighbour_indices = knn_search(
        knn_model,
        embeddings_gallery,
        labels_gallery,
        embeddings_query,
        k=k,
        exclude_self=exclude_self,
    )

    metrics = evaluate_topk(neighbour_labels, labels_query, ks=(1, k))
    metrics["mean_distance_top1"] = float(np.mean(distances[:, 0]))
    metrics["median_distance_top1"] = float(np.median(distances[:, 0]))

    metrics_path = build_output_path(output_prefix, "_knn_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Saved Deep kNN metrics: {metrics_path}")

    if visualize_count <= 0:
        return

    rng = random.Random(1337)
    indices = list(range(len(paths_query)))
    rng.shuffle(indices)
    indices = indices[:visualize_count]

    for idx_position, query_idx in enumerate(indices, start=1):
        distances_row = distances[query_idx]
        labels_row = neighbour_labels[query_idx]
        indices_row = neighbour_indices[query_idx]
        neighbour_paths = [str(paths_gallery[idx]) for idx in indices_row]

        report_path = build_output_path(output_prefix, f"_knn_example_{idx_position}.png")
        make_knn_figure(
            query_path=paths_query[query_idx],
            query_label=class_names[labels_query[query_idx]],
            neighbour_paths=neighbour_paths,
            neighbour_labels=[class_names[int(lab)] for lab in labels_row],
            neighbour_distances=distances_row,
            output_path=report_path,
        )
        print(f"Saved Deep kNN example #{idx_position}: {report_path}")


def make_knn_figure(
    query_path: Path,
    query_label: str,
    neighbour_paths: Sequence[str],
    neighbour_labels: Sequence[str],
    neighbour_distances: Sequence[float],
    output_path: Path,
) -> None:
    total = 1 + len(neighbour_paths)
    fig, axes = plt.subplots(1, total, figsize=(3 * total, 3))
    axes = np.asarray(axes).reshape(-1)

    _show_image_on_axis(axes[0], query_path, f"Query\n{query_label}")

    for ax, neigh_path, label, dist in zip(axes[1:], neighbour_paths, neighbour_labels, neighbour_distances):
        caption = f"{label}\n{dist:.3f}"
        _show_image_on_axis(ax, Path(neigh_path), caption)

    for ax in axes:
        ax.axis("off")
    fig.suptitle("Deep kNN retrieval", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _show_image_on_axis(ax, image_path: Path, title: str) -> None:
    image = Image.open(image_path).convert("RGB")
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _choose_colormap(num_classes: int):
    if num_classes <= 10:
        return plt.cm.get_cmap("tab10", num_classes)
    if num_classes <= 20:
        return plt.cm.get_cmap("tab20", num_classes)
    return plt.cm.get_cmap("gist_ncar", num_classes)


def resolve_output_prefix(args: argparse.Namespace) -> Path:
    if args.output_prefix is not None:
        prefix = args.output_prefix
        prefix.parent.mkdir(parents=True, exist_ok=True)
        return prefix

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    analysis_dir = OUTPUT_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir / f"embeddings_{args.mode}_{timestamp}"


def load_gallery_and_query(args: argparse.Namespace, seed: int):
    gallery_paths, gallery_labels, class_names = gather_image_paths(
        args.gallery_dir,
        limit_per_class=args.limit_per_class,
    )

    gallery_full_count = len(gallery_paths)
    gallery_paths, gallery_labels = sample_subset(gallery_paths, gallery_labels, args.sample_size, seed)
    if len(gallery_paths) != gallery_full_count:
        print(f"Sampled gallery set: {len(gallery_paths)} / {gallery_full_count} images")

    query_dir = args.query_dir if args.query_dir is not None else args.gallery_dir
    if query_dir == args.gallery_dir:
        query_paths = gallery_paths
        query_labels = gallery_labels
        exclude_self = True
    else:
        query_paths, query_labels, query_class_names = gather_image_paths(
            query_dir,
            limit_per_class=args.limit_per_class,
        )
        if query_class_names != class_names:
            print("Warning: class sets do not match between gallery and query; results may be skewed")
        query_full_count = len(query_paths)
        query_paths, query_labels = sample_subset(query_paths, query_labels, args.sample_size, seed + 1)
        if len(query_paths) != query_full_count:
            print(f"Sampled query set: {len(query_paths)} / {query_full_count} images")
        exclude_self = False
    return gallery_paths, gallery_labels, class_names, query_paths, query_labels, exclude_self


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_prefix = resolve_output_prefix(args)
    print(f"Outputs will be written under: {output_prefix}")

    gallery_paths, gallery_labels, class_names, query_paths, query_labels, exclude_self = load_gallery_and_query(args, args.seed)

    num_classes = len(class_names)
    print(f"Gallery images: {len(gallery_paths)} across {num_classes} identities")

    _, val_transform = get_transforms()
    gallery_dataset = ImagePathDataset(gallery_paths, gallery_labels, val_transform)
    query_dataset = ImagePathDataset(query_paths, query_labels, val_transform)

    model = load_model(args.model_path, num_classes)

    print("Extracting embeddings for gallery...")
    embeddings_gallery, labels_gallery = extract_embeddings(
        model,
        gallery_dataset,
        args.batch_size,
        args.num_workers,
        args.mode,
    )

    if query_dataset is gallery_dataset:
        embeddings_query = embeddings_gallery
        labels_query = labels_gallery
    else:
        print("Extracting embeddings for query set...")
        embeddings_query, labels_query = extract_embeddings(
            model,
            query_dataset,
            args.batch_size,
            args.num_workers,
            args.mode,
        )

    npz_path = build_output_path(output_prefix, "_embeddings.npz")
    np.savez_compressed(
        npz_path,
        embeddings_gallery=embeddings_gallery,
        labels_gallery=labels_gallery,
        paths_gallery=[str(p) for p in gallery_paths],
        embeddings_query=embeddings_query,
        labels_query=labels_query,
        paths_query=[str(p) for p in query_paths],
        class_names=np.array(class_names),
        mode=args.mode,
    )
    print(f"Saved embeddings bundle: {npz_path}")

    if args.reduction == "tsne":
        maybe_run_tsne(
            embeddings_gallery,
            labels_gallery,
            class_names,
            output_prefix,
            perplexity=args.perplexity,
            learning_rate=args.tsne_learning_rate,
            pca_components=args.pca_components,
        )

    if args.run_deepknn:
        run_deep_knn(
            embeddings_gallery,
            labels_gallery,
            gallery_paths,
            class_names,
            embeddings_query,
            labels_query,
            query_paths,
            output_prefix,
            k=args.knn_k,
            metric=args.knn_metric,
            visualize_count=args.knn_visualize_count,
            exclude_self=exclude_self,
        )

    print("Analysis complete.")


if __name__ == "__main__":
    main()
