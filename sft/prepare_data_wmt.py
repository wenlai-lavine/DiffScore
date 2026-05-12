"""
python sft/prepare_data_wmt.py \
--model_name_or_path GSAI-ML/LLaDA-8B-Base \
--sft_map_fn_path dllm.utils.default_sft_map_fn \
--lang_pairs de-en,fi-en,gu-en,kk-en,lt-en,ru-en,zh-en \
--output_dir sft/data/llada/wmt19_translation \
--max_per_pair 100000 \
--num_proc 64
"""

import importlib
import os
from dataclasses import dataclass, field
from functools import partial

import datasets
import tyro

import dllm

# ── 语言代码 → 可读名称 ──────────────────────────────────────────────
LANG_NAMES = {
    "de": "German",
    "fi": "Finnish",
    "gu": "Gujarati",
    "kk": "Kazakh",
    "lt": "Lithuanian",
    "ru": "Russian",
    "zh": "Chinese",
    "en": "English",
    "cs": "Czech",
    "fr": "French",
}


@dataclass
class ScriptArguments:
    """Preprocess SFT dataset for WMT19 machine translation."""

    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"
    sft_map_fn_path: str = "dllm.utils.default_sft_map_fn"

    # 七个语言对，逗号分隔
    lang_pairs: str = "de-en,fi-en,gu-en,kk-en,lt-en,ru-en,zh-en"
    output_dir: str = ".data/sft/llada/wmt19_translation"

    max_per_pair: int = 100_000      # 每个语言对训练集最多选取的样本数
    mask_prompt_loss: bool = True     # 用 -100 遮盖 prompt 部分的 labels
    num_proc: int = 32
    remove_columns: bool = True
    max_valid_test: int = 200         # validation / test 每个语言对最多保留数量

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


# ── 加载 & 合并多个语言对 ─────────────────────────────────────────────
def load_and_merge_translation_datasets(
    lang_pairs: str,
    num_proc: int,
    max_per_pair: int,
    max_valid_test: int,
) -> datasets.DatasetDict:
    """
    从 HuggingFace 加载 wmt/wmt19 的多个语言对，
    统一转成 messages 格式后按 split 合并。

    为避免一次性加载数百万条的大语料（如 de-en ~38M），
    使用 split 切片语法 `train[:N]` 只读取所需数量。
    """
    all_datasets_by_split: dict[str, list[datasets.Dataset]] = {}
    pairs = [p.strip() for p in lang_pairs.split(",")]

    for pair in pairs:
        src, tgt = pair.split("-")
        src_name = LANG_NAMES.get(src, src)
        tgt_name = LANG_NAMES.get(tgt, tgt)

        print(f"\n{'='*60}")
        print(f"Loading WMT19  {pair}  ({src_name} → {tgt_name})")
        print(f"{'='*60}")

        # 翻译 prompt 模板
        prompt_template = (
            f"Translate the following text from {src_name} to {tgt_name}.\n\n"
            f"{{text}}"
        )

        # ---- 按 split 加载 ------------------------------------------------
        split_limits = {
            "train": max_per_pair,
            "validation": max_valid_test,
            # "test": max_valid_test,
        }

        for split_name, limit in split_limits.items():
            # 利用 split 切片语法高效加载（不会把整个大数据集读进内存）
            split_slice = f"{split_name}[:{limit}]"
            try:
                split_ds = datasets.load_dataset(
                    "wmt/wmt19", pair, split=split_slice, trust_remote_code=True
                )
            except Exception as e:
                # 有些语言对没有 test split，或数据量不足 limit 条
                try:
                    split_ds = datasets.load_dataset(
                        "wmt/wmt19", pair, split=split_name, trust_remote_code=True
                    )
                except Exception:
                    print(f"  ⚠ {split_name} split not available for {pair}, skipping.")
                    continue

            print(f"  {split_name}: loaded {len(split_ds)} examples")

            # ---- 转换为 messages 格式 ------------------------------------
            # 注意：闭包要用默认参数绑定当前循环的变量
            def convert_to_messages(
                example,
                _src=src,
                _tgt=tgt,
                _tmpl=prompt_template,
            ):
                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": _tmpl.format(
                                text=example["translation"][_src]
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": example["translation"][_tgt],
                        },
                    ]
                }

            original_columns = split_ds.column_names
            split_ds = split_ds.map(
                convert_to_messages,
                num_proc=num_proc,
                remove_columns=original_columns,
                desc=f"Converting {pair} / {split_name}",
            )

            all_datasets_by_split.setdefault(split_name, []).append(split_ds)

    # ---- 合并所有语言对 ---------------------------------------------------
    print(f"\n{'='*60}")
    print("Concatenating all language pairs …")
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

    # 只保留训练必需的列以节省磁盘空间
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

    # 加载 WMT19 翻译数据并转为统一格式
    dataset = load_and_merge_translation_datasets(
        lang_pairs=args.lang_pairs,
        num_proc=args.num_proc,
        max_per_pair=args.max_per_pair,
        max_valid_test=args.max_valid_test,
    )

    # 动态导入 sft_map_fn
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