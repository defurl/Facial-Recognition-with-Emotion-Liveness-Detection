"""Employee database utilities: load, save, and build embedding index."""
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.config import DEVICE, EMPLOYEE_DB_PATH, USE_MULTI_EMBEDDING


def load_employee_db(db_path: Path = EMPLOYEE_DB_PATH) -> Dict[str, torch.Tensor]:
    """Load employee database from disk (returns empty dict if missing)."""
    if db_path.exists():
        db = torch.load(db_path, map_location="cpu")
        if not isinstance(db, dict):
            raise ValueError(f"Invalid employee DB format at {db_path}")
        return db
    return {}


def save_employee_db(db: Dict[str, torch.Tensor], db_path: Path = EMPLOYEE_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(db, db_path)


def build_embedding_index(
    employee_db: Dict[str, torch.Tensor],
    device=DEVICE,
    use_multi_embedding: bool = USE_MULTI_EMBEDDING,
) -> Tuple[torch.Tensor, List[Tuple[str, int, int]]]:
    """
    Build a contiguous tensor of all embeddings and offsets per identity.

    Returns (emb_tensor, offsets) where offsets is list of (name, start, end).
    If DB is empty, returns (None, []).
    """
    embeddings: List[torch.Tensor] = []
    offsets: List[Tuple[str, int, int]] = []
    cursor = 0

    for name, stored in employee_db.items():
        if use_multi_embedding and isinstance(stored, list):
            emb_list = [emb.detach().float().cpu() for emb in stored]
        else:
            emb = stored[0] if isinstance(stored, list) else stored
            emb_list = [emb.detach().float().cpu()]

        if len(emb_list) == 0:
            continue

        stacked = torch.stack(emb_list)
        count = stacked.shape[0]
        embeddings.append(stacked)
        offsets.append((name, cursor, cursor + count))
        cursor += count

    if not embeddings:
        return None, []

    all_emb = torch.cat(embeddings, dim=0).to(device, non_blocking=True)
    return all_emb, offsets
