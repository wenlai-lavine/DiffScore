"""
python sft/prepare_data.py \
--model_name_or_path GSAI-ML/LLaDA-8B-Base \
--sft_map_fn_path dllm.utils.default_sft_map_fn \
--dataset_names cnn_dailymail,xsum \
--output_dir sft/data/llada/cnn_xsum \
--num_proc 64
"""

import importlib
import os
from dataclasses import dataclass, field
from functools import partial

import datasets
import tyro

import dllm

@dataclass
class ScriptArguments:
    """Preprocess SFT dataset."""

    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"
    sft_map_fn_path: str = "dllm.utils.default_sft_map_fn"
    
    # 1. 改为列表，支持传入多个数据集名称
    dataset_names: str = "cnn_dailymail,xsum" 
    output_dir: str = ".data/sft/llada/summarization_merged"
    
    mask_prompt_loss: bool = True  # Mask prompt tokens in labels with -100
    num_proc: int = 32
    remove_columns: bool = True
    max_valid_test: int = 500  # 新增参数，控制validation和test的最大数据量

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


def load_and_merge_summarization_datasets(dataset_names: str, num_proc: int, max_valid_test: int) -> datasets.DatasetDict:
    """
    加载多个 summarization 数据集，统一转成 messages 格式并合并
    """
    all_datasets_by_split = {}
    
    # 构建 Prompt 的模板
    prompt_template = "Summarize the following text:\n\n{text}"
    dataset_names = dataset_names.split(",")
    for ds_name in dataset_names:
        print(f"Loading and formatting dataset: {ds_name}...")
        
        # 针对不同的数据集设定特定的 config 和 字段名
        if "cnn_dailymail" in ds_name:
            ds = datasets.load_dataset("cnn_dailymail", "3.0.0")
            source_col, target_col = "article", "highlights"
        elif "xsum" in ds_name:
            ds = datasets.load_dataset("xsum")
            source_col, target_col = "document", "summary"
        else:
            # 兼容其他数据集的默认逻辑（如有需要可自行拓展）
            ds = datasets.load_dataset(ds_name)
            source_col, target_col = "text", "summary" 
            
        # 定义转换函数，将摘要数据转换为原程序支持的 messages 格式
        def convert_to_messages(example):
            return {
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt_template.format(text=example[source_col])
                    },
                    {
                        "role": "assistant", 
                        "content": example[target_col]
                    }
                ]
            }

        # 映射转换，并删除旧列（只保留 messages）
        original_columns = ds["train"].column_names
        ds = ds.map(
            convert_to_messages,
            num_proc=num_proc,
            remove_columns=original_columns, 
            desc=f"Converting {ds_name} to messages format"
        )

        # 按 split (train/validation/test) 收集数据集
        for split in ds.keys():
            if split not in all_datasets_by_split:
                all_datasets_by_split[split] = []
            # 如果是validation或test集，限制最大数量
            if split in ["validation", "test"]:
                all_datasets_by_split[split].append(ds[split].select(range(min(len(ds[split]), max_valid_test))))
            else:
                all_datasets_by_split[split].append(ds[split])
        
    # 合并（Concatenate）所有相同 split 的数据集
    print("Concatenating all datasets...")
    merged_dict = {
        split: datasets.concatenate_datasets(dsets)
        for split, dsets in all_datasets_by_split.items()
    }
    
    return datasets.DatasetDict(merged_dict)


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

    # Keep only the three required columns to save space.
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
    print(f"[OK] Saved to: {output_dir}")


def main():
    # Parse with tyro
    args = tyro.cli(ScriptArguments)
    dllm.utils.print_args(args)

    tokenizer = dllm.utils.get_tokenizer(args)

    # 2. 替换原始的数据加载逻辑，使用我们自定义的多数据集加载并转格式的方法
    dataset = load_and_merge_summarization_datasets(
        dataset_names=args.dataset_names,
        num_proc=args.num_proc,
        max_valid_test=args.max_valid_test
    )

    # 4. Dynamically import the function based on the argument
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