"""Phase 1 — Data preparation.

Downloads MMSearch-Plus from HuggingFace (Cie1/MMSearch-Plus) and applies the
official XOR+SHA-256 text decryption (canary='MMSearch-Plus'). Per the official
decrypt_after_load.py, images in the current release are NOT encrypted.

Outputs:
  - data/raw/mmsearch_plus_full.jsonl     (all 311 tasks, decrypted text fields)
  - data/raw/images/<task_id>/img_{k}.png (images per task, paths relative to project root)
  - data/raw/mmsearch_plus_sample.jsonl   (stratified sample of N tasks)

Sampling: stratified on `category × difficulty`; seed recorded in sample header.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


CANARY = "MMSearch-Plus"
REPO_ID = "Cie1/MMSearch-Plus"
OUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


def _derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode()).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def _decrypt_text(ciphertext_b64: str, password: str) -> str:
    if not ciphertext_b64:
        return ciphertext_b64
    try:
        encrypted = base64.b64decode(ciphertext_b64)
        key = _derive_key(password, len(encrypted))
        return bytes(a ^ b for a, b in zip(encrypted, key)).decode("utf-8")
    except Exception:
        return ciphertext_b64


def _save_image(blob: Any, out_path: Path) -> bool:
    if blob is None:
        return False
    img: Image.Image | None = None
    if isinstance(blob, Image.Image):
        img = blob
    elif isinstance(blob, dict):
        if blob.get("bytes"):
            img = Image.open(io.BytesIO(blob["bytes"]))
        elif blob.get("path"):
            img = Image.open(blob["path"])
    if img is None:
        return False
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    img.save(out_path, format="PNG")
    return True


def load_and_decrypt() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(REPO_ID, split="train")
    print(f"Loaded {len(ds)} raw samples. Decrypting text + saving images…")
    out: list[dict] = []
    for i, row in enumerate(ds):
        task_id = f"mmsp_{i:04d}"
        rec: dict = {
            "task_id": task_id,
            "question": _decrypt_text(row.get("question", ""), CANARY),
            "answer": [_decrypt_text(a, CANARY) for a in (row.get("answer") or [])],
            "num_images": int(row.get("num_images") or 0),
            "arxiv_id": _decrypt_text(row.get("arxiv_id") or "", CANARY) or None,
            "video_url": _decrypt_text(row.get("video_url") or "", CANARY) or None,
            "category": row.get("category"),
            "difficulty": row.get("difficulty"),
            "subtask": row.get("subtask"),
            "image_paths": [],
        }
        img_dir = OUT_ROOT / "images" / task_id
        img_dir.mkdir(parents=True, exist_ok=True)
        for k in range(1, 6):
            blob = row.get(f"img_{k}")
            out_p = img_dir / f"img_{k}.png"
            if _save_image(blob, out_p):
                rec["image_paths"].append(str(out_p.relative_to(OUT_ROOT.parent.parent)))
        out.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ds)} processed")
    return out


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        buckets[(r.get("category") or "?", r.get("difficulty") or "?")].append(r)
    rng = random.Random(seed)
    total = len(records)
    picked: list[dict] = []
    remainders: list[tuple[float, list[dict]]] = []
    for _key, bucket in buckets.items():
        rng.shuffle(bucket)
        quota_f = n * len(bucket) / total
        quota = int(quota_f)
        picked.extend(bucket[:quota])
        if quota < len(bucket):
            remainders.append((quota_f - quota, bucket[quota:]))
    remainders.sort(key=lambda x: -x[0])
    for _frac, rem in remainders:
        if len(picked) >= n:
            break
        picked.append(rem[0])
    rng.shuffle(picked)
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260417)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    full = load_and_decrypt()

    full_path = OUT_ROOT / "mmsearch_plus_full.jsonl"
    with full_path.open("w") as f:
        for r in full:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(full)} -> {full_path}")

    print("\nOverall distribution:")
    cat_all = defaultdict(int)
    diff_all = defaultdict(int)
    for r in full:
        cat_all[r.get("category") or "?"] += 1
        diff_all[r.get("difficulty") or "?"] += 1
    print(f"  category: {dict(cat_all)}")
    print(f"  difficulty: {dict(diff_all)}")

    sample = stratified_sample(full, n=min(args.n, len(full)), seed=args.seed)
    sample_path = OUT_ROOT / "mmsearch_plus_sample.jsonl"
    with sample_path.open("w") as f:
        f.write(json.dumps({"_meta": {"seed": args.seed, "size": len(sample),
                                       "source": REPO_ID,
                                       "strata": "category×difficulty"}}) + "\n")
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote sample({len(sample)}) -> {sample_path}")

    cats = defaultdict(int)
    diffs = defaultdict(int)
    for r in sample:
        cats[r.get("category") or "?"] += 1
        diffs[r.get("difficulty") or "?"] += 1
    print("Sample distribution:")
    print(f"  category: {dict(cats)}")
    print(f"  difficulty: {dict(diffs)}")


if __name__ == "__main__":
    main()
