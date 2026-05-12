"""
python sft/prepare_data_d2t.py \
--model_name_or_path GSAI-ML/LLaDA-8B-Base \
--sft_map_fn_path dllm.utils.default_sft_map_fn \
--dataset_names totto,e2e_nlg,web_nlg \
--output_dir .data/sft/llada/d2t_merged \
--max_per_dataset 100000 \
--num_proc 64
"""

import importlib
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List

import datasets
import tyro

import dllm


# ── 将各数据集的结构化输入线性化为纯文本 ──────────────────────────────────


def linearize_totto_table(example: Dict[str, Any]) -> str:
    """将 ToTTo 的 table + highlighted_cells 转为可读文本。

    格式:
        Page: {title} | Section: {section}
        [H] Col1 : Val1 | Col2 : Val2 | ...
        (未高亮的行被省略，仅保留高亮单元格所在行的信息)
    """
    title = example.get("table_page_title", "")
    section = example.get("table_section_title", "")
    table = example.get("table", [])
    highlighted = example.get("highlighted_cells", [])

    if not table or not highlighted:
        return f"Page: {title} | Section: {section}"

    highlighted_set = {(r, c) for r, c in highlighted}

    header_row = table[0] if table else []
    header_vals = [cell["value"] for cell in header_row]

    parts = []
    if title:
        parts.append(f"Page: {title}")
    if section:
        parts.append(f"Section: {section}")

    cell_descs: List[str] = []
    for row_idx, col_idx in sorted(highlighted_set):
        if row_idx >= len(table) or col_idx >= len(table[row_idx]):
            continue
        cell_val = table[row_idx][col_idx]["value"]
        col_name = header_vals[col_idx] if col_idx < len(header_vals) else f"Col{col_idx}"
        if table[row_idx][col_idx].get("is_header", False):
            continue
        cell_descs.append(f"{col_name} : {cell_val}")

    if cell_descs:
        parts.append("Highlighted cells: " + " | ".join(cell_descs))

    return "\n".join(parts) if parts else title


def linearize_webnlg_triples(example: Dict[str, Any]) -> str:
    """将 WebNLG 的 RDF triple 列表拼接为文本。

    input 字段为 List[str]，每项格式 "subject | predicate | object"。
    """
    triples = example.get("input", [])
    if isinstance(triples, list):
        return " ; ".join(triples)
    return str(triples)


def linearize_e2e(example: Dict[str, Any]) -> str:
    """E2E NLG 的 meaning_representation 已经是文本，直接返回。"""
    return example.get("meaning_representation", "")


# ── 获取目标文本 ────────────────────────────────────────────────────────


def get_target_text(example: Dict[str, Any], ds_name: str) -> str:
    """从不同数据集中提取目标文本。"""
    if "target" in example and example["target"]:
        return example["target"]
    if ds_name == "totto":
        annots = example.get("sentence_annotations", {})
        if isinstance(annots, dict):
            return annots.get("final_sentence", "")
        if isinstance(annots, list) and annots:
            return annots[0].get("final_sentence", "")
    return ""


# ── 数据集名称 → 加载/线性化策略 ────────────────────────────────────────

DATASET_CONFIG = {
    "totto": {
        "hf_name": "GEM/totto",
        "linearize_fn": linearize_totto_table,
    },
    "e2e_nlg": {
        "hf_name": "GEM/e2e_nlg",
        "linearize_fn": linearize_e2e,
    },
    "web_nlg": {
        "hf_name": "GEM/web_nlg",
        "hf_config": "en",
        "linearize_fn": linearize_webnlg_triples,
    },
}


@dataclass
class ScriptArguments:
    """Preprocess SFT dataset for Data-to-Text (D2T) generation."""

    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"
    sft_map_fn_path: str = "dllm.utils.default_sft_map_fn"

    dataset_names: str = "totto,e2e_nlg,web_nlg"
    output_dir: str = ".data/sft/llada/d2t_merged"

    max_per_dataset: int = 100_000
    mask_prompt_loss: bool = True
    num_proc: int = 32
    remove_columns: bool = True
    max_valid_test: int = 500

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


# ── 加载 & 合并多个 D2T 数据集 ──────────────────────────────────────────

PROMPT_TEMPLATE = (
    "Generate a natural language description for the following structured data:"
    "\n\n{data}"
)


def load_and_merge_d2t_datasets(
    dataset_names: str,
    num_proc: int,
    max_per_dataset: int,
    max_valid_test: int,
) -> datasets.DatasetDict:
    """
    加载 GEM 中的多个 D2T 数据集，统一转成 messages 格式后按 split 合并。
    """
    all_datasets_by_split: dict[str, list[datasets.Dataset]] = {}
    names = [n.strip() for n in dataset_names.split(",")]

    for ds_name in names:
        cfg = DATASET_CONFIG.get(ds_name)
        if cfg is None:
            print(f"  Warning: unknown dataset '{ds_name}', skipping.")
            continue

        hf_name = cfg["hf_name"]
        hf_config = cfg.get("hf_config", None)
        linearize_fn = cfg["linearize_fn"]

        print(f"\n{'='*60}")
        print(f"Loading {hf_name}" + (f" (config={hf_config})" if hf_config else ""))
        print(f"{'='*60}")

        try:
            if hf_config:
                ds = datasets.load_dataset(hf_name, hf_config, trust_remote_code=True)
            else:
                ds = datasets.load_dataset(hf_name, trust_remote_code=True)
        except Exception as e:
            print(f"  Error loading {hf_name}: {e}")
            continue

        split_limits = {
            "train": max_per_dataset,
            "validation": max_valid_test,
            "test": max_valid_test,
        }

        for split_name, limit in split_limits.items():
            if split_name not in ds:
                print(f"  {split_name} split not available, skipping.")
                continue

            split_ds = ds[split_name]
            if len(split_ds) > limit:
                split_ds = split_ds.select(range(limit))

            print(f"  {split_name}: {len(split_ds)} examples")

            def convert_to_messages(
                example,
                _lin_fn=linearize_fn,
                _ds_name=ds_name,
            ):
                source_text = _lin_fn(example)
                target_text = get_target_text(example, _ds_name)
                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": PROMPT_TEMPLATE.format(data=source_text),
                        },
                        {
                            "role": "assistant",
                            "content": target_text,
                        },
                    ]
                }

            original_columns = split_ds.column_names
            split_ds = split_ds.map(
                convert_to_messages,
                num_proc=num_proc,
                remove_columns=original_columns,
                desc=f"Converting {ds_name} / {split_name}",
            )

            split_ds = split_ds.filter(
                lambda x: bool(x["messages"][1]["content"]),
                num_proc=num_proc,
                desc=f"Filtering empty targets in {ds_name} / {split_name}",
            )

            all_datasets_by_split.setdefault(split_name, []).append(split_ds)

    print(f"\n{'='*60}")
    print("Concatenating all D2T datasets ...")
    merged_dict = {
        split: datasets.concatenate_datasets(ds_list)
        for split, ds_list in all_datasets_by_split.items()
    }
    for split, ds in merged_dict.items():
        print(f"  {split}: {len(ds)} total examples")

    return datasets.DatasetDict(merged_dict)


# ── 预处理（tokenize + 保存到磁盘）─────────────────────────────────────

def preprocess_sft_dataset(
    dataset: datasets.DatasetDict,
    map_fn: callable,
    output_dir: str,
    remove_columns: bool = False,
    num_proc: int = 32,
):
    processed = dataset.map(
        map_fn,
        batched=False,
        num_proc=num_proc,
        load_from_cache_file=True,
        writer_batch_size=512,
        desc="offline preprocessing",
    )

    if remove_columns:
        keep = {"input_ids", "labels", "prompt_len", "attention_mask"}

        def strip_cols(ds: datasets.Dataset) -> datasets.Dataset:
            drop = [c for c in ds.column_names if c not in keep]
            return ds.remove_columns(drop) if drop else ds

        if isinstance(processed, datasets.DatasetDict):
            for split in list(processed.keys()):
                processed[split] = strip_cols(processed[split])
        else:
            processed = strip_cols(processed)

    os.makedirs(output_dir, exist_ok=True)
    processed.save_to_disk(output_dir)
    print(f"\n[OK] Saved to: {output_dir}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    args = tyro.cli(ScriptArguments)
    dllm.utils.print_args(args)

    tokenizer = dllm.utils.get_tokenizer(args)

    dataset = load_and_merge_d2t_datasets(
        dataset_names=args.dataset_names,
        num_proc=args.num_proc,
        max_per_dataset=args.max_per_dataset,
        max_valid_test=args.max_valid_test,
    )

    try:
        module_path, function_name = args.sft_map_fn_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        sft_map_fn = getattr(module, function_name)
    except (ImportError, AttributeError, ValueError) as e:
        print(f"Error: Could not import '{args.sft_map_fn_path}'.")
        print(f"Details: {e}")
        return

    map_fn = partial(
        sft_map_fn,
        tokenizer=tokenizer,
        mask_prompt_loss=args.mask_prompt_loss,
    )

    preprocess_sft_dataset(
        dataset=dataset,
        map_fn=map_fn,
        output_dir=args.output_dir,
        remove_columns=args.remove_columns,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()
