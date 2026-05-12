"""
Merge preprocessed SUM / WMT / D2T SFT datasets (saved with save_to_disk) into one
DatasetDict for multi-task MDLM SFT training.

Example (paths match defaults from prepare_data_sum.py / prepare_data_wmt.py / prepare_data_d2t.py):

python sft/prepare_data_merge_multitask.py \
  --sum_path .data/sft/llada/summarization_merged \
  --wmt_path .data/sft/llada/wmt19_translation \
  --d2t_path .data/sft/llada/d2t_merged \
  --output_dir .data/sft/llada/multitask_sum_wmt_d2t \
  --shuffle_train True \
  --seed 42
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import datasets
import tyro


def _assert_compatible_columns(
    parts: list[tuple[str, datasets.Dataset]], split: str
) -> None:
    """All shards for a split must share the same column names and dtypes."""
    if not parts:
        return
    ref_cols = parts[0][1].column_names
    ref_features = parts[0][1].features
    for name, ds in parts[1:]:
        if ds.column_names != ref_cols:
            raise ValueError(
                f"Column mismatch on split '{split}': "
                f"first has {ref_cols}, but '{name}' has {ds.column_names}"
            )
        if ds.features != ref_features:
            raise ValueError(
                f"Feature mismatch on split '{split}' for '{name}' vs first shard."
            )


def _maybe_cap(
    ds: datasets.Dataset,
    max_rows: int | None,
) -> datasets.Dataset:
    if max_rows is None or len(ds) <= max_rows:
        return ds
    return ds.select(range(max_rows))


def merge_multitask(
    sum_ds: datasets.DatasetDict,
    wmt_ds: datasets.DatasetDict,
    d2t_ds: datasets.DatasetDict,
    *,
    cap_sum_train: int | None = None,
    cap_wmt_train: int | None = None,
    cap_d2t_train: int | None = None,
    cap_sum_eval: int | None = None,
    cap_wmt_eval: int | None = None,
    cap_d2t_eval: int | None = None,
    shuffle_train: bool = False,
    seed: int = 42,
) -> datasets.DatasetDict:
    """
    Concatenate per split. Splits are the union of keys across the three dicts;
    a task that lacks a split is simply omitted for that split.
    """
    names_and_dicts: list[tuple[str, datasets.DatasetDict]] = [
        ("sum", sum_ds),
        ("wmt", wmt_ds),
        ("d2t", d2t_ds),
    ]

    train_caps = {
        "sum": cap_sum_train,
        "wmt": cap_wmt_train,
        "d2t": cap_d2t_train,
    }
    eval_caps = {
        "sum": cap_sum_eval,
        "wmt": cap_wmt_eval,
        "d2t": cap_d2t_eval,
    }

    all_splits: set[str] = set()
    for _, d in names_and_dicts:
        all_splits.update(d.keys())

    out: dict[str, datasets.Dataset] = {}

    for split in sorted(all_splits):
        parts: list[tuple[str, datasets.Dataset]] = []
        is_train = split == "train"
        for task_name, d in names_and_dicts:
            if split not in d:
                continue
            shard = d[split]
            cap = train_caps[task_name] if is_train else eval_caps[task_name]
            shard = _maybe_cap(shard, cap)
            parts.append((task_name, shard))

        if not parts:
            continue

        _assert_compatible_columns(parts, split)
        merged = datasets.concatenate_datasets([p[1] for p in parts])

        if split == "train" and shuffle_train:
            merged = merged.shuffle(seed=seed)

        out[split] = merged

    return datasets.DatasetDict(out)


@dataclass
class ScriptArguments:
    """Merge three on-disk SFT datasets (SUM, WMT, D2T) for multi-task training."""

    sum_path: str = ".data/sft/llada/summarization_merged"
    wmt_path: str = ".data/sft/llada/wmt19_translation"
    d2t_path: str = ".data/sft/llada/d2t_merged"

    output_dir: str = ".data/sft/llada/multitask_sum_wmt_d2t"

    # Optional caps (per task, before concatenation). None = use full split.
    cap_sum_train: int | None = None
    cap_wmt_train: int | None = None
    cap_d2t_train: int | None = None
    cap_sum_eval: int | None = None
    cap_wmt_eval: int | None = None
    cap_d2t_eval: int | None = None

    shuffle_train: bool = True
    seed: int = 42


def _load(path: str, label: str) -> datasets.DatasetDict:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{label} dataset path does not exist or is not a directory: {path}"
        )
    ds = datasets.load_from_disk(path)
    if not isinstance(ds, datasets.DatasetDict):
        raise ValueError(
            f"Expected DatasetDict at {path}, got {type(ds).__name__}"
        )
    return ds


def main() -> None:
    args = tyro.cli(ScriptArguments)

    print("Loading SUM …")
    sum_ds = _load(args.sum_path, "SUM")
    print("Loading WMT …")
    wmt_ds = _load(args.wmt_path, "WMT")
    print("Loading D2T …")
    d2t_ds = _load(args.d2t_path, "D2T")

    for label, ds in [("SUM", sum_ds), ("WMT", wmt_ds), ("D2T", d2t_ds)]:
        print(f"  {label} splits: {list(ds.keys())}")
        for sp, sub in ds.items():
            print(f"    {sp}: {len(sub)} rows")

    merged = merge_multitask(
        sum_ds,
        wmt_ds,
        d2t_ds,
        cap_sum_train=args.cap_sum_train,
        cap_wmt_train=args.cap_wmt_train,
        cap_d2t_train=args.cap_d2t_train,
        cap_sum_eval=args.cap_sum_eval,
        cap_wmt_eval=args.cap_wmt_eval,
        cap_d2t_eval=args.cap_d2t_eval,
        shuffle_train=args.shuffle_train,
        seed=args.seed,
    )

    print("\nMerged dataset:")
    for sp, sub in merged.items():
        print(f"  {sp}: {len(sub)} rows")

    os.makedirs(os.path.dirname(args.output_dir) or ".", exist_ok=True)
    merged.save_to_disk(args.output_dir)
    print(f"\n[OK] Saved merged multitask data to: {args.output_dir}")


if __name__ == "__main__":
    main()
