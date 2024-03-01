# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
import torch
from pathlib import Path

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List, Tuple
import time
import gact
from gact.utils import get_memory_usage, get_weight_memory, get_gradient_memory, get_optimizer_memory, exp_recorder
from gact.flops_ops import ModuleFLOPs_Linear, ModuleFLOPs_Norm, ModuleFLOPs_GELU, ModuleFLOPs_QK, ModuleFLOPs_OV, MethodFLOPs_softmax_from_Q
from utils import AverageMeter
import wandb

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

import peft
from peft import get_peft_model, LoraConfig, PeftType, PeftModelForSequenceClassification

import bitsandbytes as bnb
from lpmm import optim

from gact.controller import Controller
from gact.efficient_linear import EfficientMemoryLinear

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

metric_key = {
    'mrpc': 'f1',
    'sst2': 'accuracy',
    'mnli': 'accuracy',
    'qnli': 'accuracy',
    'rte': 'accuracy',
    'qqp': 'f1',
    'stsb': 'pearsonr',
    'cola': 'matthews_correlation',
    'wnli': 'accuracy',
}

flops_intensity_dict = {
    'gemm': 2,
    'norm': 5,
    'softmax': 10,
    'actfn': 10
}

total_flops = 0
total_flops_intensity = 0
total_forward_flops = 0
total_backward_flops = 0

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_gradient_norm",
        type=float,
        default=1.,
        help="Maximum norm of gradient.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--get-mem", action="store_true", help="Whether or not to check the usage of memory")
    parser.add_argument("--get-speed", action="store_true", help="Whether or not to get ips")
    parser.add_argument("--gact", action="store_true", help="Whether or not to use gact")
    parser.add_argument("--opt_level", type=str, help="Optimization level of gact")
    parser.add_argument("--get-macs", action="store_true", help="Get Number of Macs")
    parser.add_argument('--customize', action='store_true')
    parser.add_argument("--layer_num", type=int, default=24, help="Number of Bert layers")
    parser.add_argument("--hidden_size", type=int, default=1024, help="hidden size")
    parser.add_argument("--intermediate_size", type=int, default=4096, help='customize intermediate size')
    parser.add_argument("--ckpt", action='store_true', help='enable gradient checkpoint')
    # config about lora
    parser.add_argument("--lora", action='store_true', help='enable lora')
    parser.add_argument("--lora-all-linears", action='store_true', help='lora all linears')
    parser.add_argument("--use-fp4", action='store_true', help='use fp4')
    parser.add_argument("--r", action='store_true', help='lora rank', default=8)
    # config about optimizer
    parser.add_argument("--optimizer-8bit", action='store_true', help='use 8bit optimizer')
    parser.add_argument("--optimizer-4bit", action='store_true', help='use 4bit optimizer')
    # config about sparse bp
    parser.add_argument("--sparse-bp", action='store_true', help='enable sparse bp')
    parser.add_argument("--sparse-bp-freeze-range", type=str, default="[i for i in range(12)]", help='sparse bp range')
    parser.add_argument("--sparse-bp-freeze-layer", type=str, default="[]", help='sparse bp layer')
    # config about linear probe
    parser.add_argument("--linear-probe", action='store_true', help='enable linear probe')

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    if args.gact:
        assert args.gradient_accumulation_steps == 1, "gact works with accumulation step = 1"

    return args


def find_all_linear_names(args, model):
    # cls = bnb.nn.Linear4bit if args.use_fp4 == 4 else torch.nn.Linear
    # lora_module_names = set()
    # for name, module in model.named_modules():
    #     if isinstance(module, cls):
    #         names = name.split('.')
    #         lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # if 'out_proj' in lora_module_names: # needed for 16-bit
    #     lora_module_names.remove('out_proj')
    # print(lora_module_names)
    return ['dense', 'key', 'value', 'query']


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(device_placement=False)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("/home/yujin-wa20/projects/GACT-ICML/benchmark/text_classification/glue.py", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)

    if args.use_fp4:
        assert args.lora, "use fp4 must use lora"
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path, 
            config=config,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16, 
                bnb_4bit_use_double_quant=True, 
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["classifier"]
            ),
            ignore_mismatched_sizes=True, 
            device_map="auto",
        )
        use_gradient_checkpointing = args.ckpt
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

        if args.ckpt:
            model.gradient_checkpointing_enable()

    print(model)

    if args.lora:
        peft_config = LoraConfig(
            task_type="SEQ_CLS", 
            inference_mode=False, 
            r=args.r, 
            lora_alpha=16, 
            lora_dropout=0.1,
            target_modules=find_all_linear_names(args, model),
        )   
        model = PeftModelForSequenceClassification(model, peft_config)

    if args.sparse_bp:
        # freeze the layers
        print(f"Freeze range: {args.sparse_bp_freeze_range}")
        for name, param in model.named_parameters():
            print(name)
            if len(name.split('.')) > 3 and name.split('.')[3].isdigit() and int(name.split('.')[3]) in eval(args.sparse_bp_freeze_range): # freeze certain layers
                param.requires_grad = False
            if 'embeddings' in name:
                param.requires_grad = False
            for item in eval(args.sparse_bp_freeze_layer): # freeze certain parts
                if item in name:
                    param.requires_grad = False

    if args.linear_probe:
        # only train the classifier
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    
    print(model)

    # print trainable parameter ratio:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params
    print(f"Total params: {total_params}, Trainable params: {trainable_params}, Trainable ratio: {trainable_ratio}")

    if args.gact:
        gact.set_optimization_level(args.opt_level)
        controller = Controller(model)

    if 'JPEG' in args.opt_level or 'DCT' in args.opt_level: #! use another method -- linear replace
        # replace all the linear layers with compressed based linear layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                original_weight_data = module.weight.data
                original_bias_data = module.bias.data if module.bias is not None else None
                new_module = EfficientMemoryLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    compress_type='JPEG' if 'JPEG' in args.opt_level else 'DCT',
                    compress_quality=int(args.opt_level.split('+')[-1]),
                )
                new_module.weight.data = original_weight_data
                if module.bias is not None:
                    new_module.bias.data = original_bias_data

    # update optimization level, this is only for logging output
    if args.ckpt:
        args.opt_level += '_ckpt'
            
    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    train_max_length = 0
    dev_max_length = 0
    for item in train_dataset:
        if len(item['input_ids']) > train_max_length:
            train_max_length = len(item['input_ids'])
    for item in eval_dataset:
        if len(item['input_ids']) > dev_max_length:
            dev_max_length = len(item['input_ids'])
    logger.info('Train max length: %d' % train_max_length)
    logger.info('Dev max length: %d' % dev_max_length)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    assert not (args.optimizer_8bit and args.optimizer_4bit), "8bit and 4bit cannot be used at the same time"
    if args.optimizer_8bit:
        optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_4bit:
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    total_mem = AverageMeter('Total Memory', ':.4e')
    peak_mem = AverageMeter('Peak Memory', ':.4e')
    activation_mem = AverageMeter('Activation Memory', ':.4e')
    init_mem = AverageMeter('Init Memory', ':.4e')

    def pack_hook(tensor): # quantize hook
        if args.gact:
            return controller.quantize(tensor)
        return tensor

    def unpack_hook(tensor): # dequantize hook
        if args.gact:
            return controller.dequantize(tensor)
        return tensor


    iter = 0
    train_step = 0
    val_step = 0
    best_metric = 0
    batch_total_time = 0
    model.to(args.device)

    # define wandb logger
    wandb_project = f"{args.model_name_or_path.split('/')[-1]}_{args.task_name}"
    wandb_name = f"lr_{args.learning_rate}_{args.opt_level}_lora_{args.lora}_low-bit-optimizer_{args.optimizer_4bit}_sparse-bp_{args.sparse_bp}_sparse-bp-range_{args.sparse_bp_freeze_range}_sparse-bp-layers_{args.sparse_bp_freeze_layer}_linear-probe_{args.linear_probe}_30"
    wandb.init(project=wandb_project, name=wandb_name)
    wandb.define_metric("train_step")
    wandb.define_metric("val_step")
    wandb.define_metric("train_*", step_metric="train_step")
    wandb.define_metric("val_*", step_metric="val_step")
    wandb.log({"trainable_ratio": trainable_ratio})

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                iter += 1
                train_step += 1
        
                if args.get_mem and iter > 1:
                    torch.cuda.synchronize()
                    # accelerator.print("===============After Data Loading=======================")
                    init_mem = get_memory_usage(False)  # model size + data size
                    torch.cuda.reset_peak_memory_stats()
                
                if args.get_speed and iter > 1:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
        
                    torch.cuda.synchronize()
                    start.record()
                
                for k, v in batch.items():
                    batch[k] = v.to(args.device)
                
                if args.get_macs:

                    def flops_forward_hook(module, inputs, outputs):
                        global total_flops, total_flops_intensity, total_forward_flops, total_backward_flops
                        class_type = module.__class__.__name__
                        # # only for debug
                        # print(f"forward: {module.name}, {module.__class__.__name__}")

                        if class_type == 'Linear': # y = x @ W + b, (B,M,K) @ (K,N) + (N,) = (B,M,N)
                            if (not args.lora) or (args.lora and "base_layer" not in module.name):
                                flops = ModuleFLOPs_Linear(module, outputs, inputs[0])
                                total_flops += flops
                                total_forward_flops += flops
                                module.flops = flops # save the flops for backward

                        elif class_type == 'LayerNorm':
                            flops = ModuleFLOPs_Norm(module, outputs, inputs[0])
                            total_flops += flops
                            total_forward_flops += flops
                            module.flops = flops

                        elif class_type == 'GELUActivation' or class_type == 'SiLUActivation': # GELU, usually used in BERT type models
                            flops = ModuleFLOPs_GELU(module, outputs, inputs[0])
                            total_flops += flops
                            total_forward_flops += flops
                            module.flops = flops

                        if 'query' in module.name: # specially for (Q @ K.T) @ V
                            # Q @ K
                            flops = ModuleFLOPs_QK(outputs)
                            # softmax
                            flops += MethodFLOPs_softmax_from_Q(outputs)
                            # O @ V
                            flops += ModuleFLOPs_OV(outputs)
                            total_flops += flops
                            total_forward_flops += flops
                            module.attn_out_shape = outputs.shape
                    
                    
                    def flops_backward_hook(module, grad_input, grad_output):
                        global total_flops, total_flops_intensity, total_forward_flops, total_backward_flops
                        class_type = module.__class__.__name__
                        # # only for debug
                        # print(f"backward: {module.name}, {module.__class__.__name__}")

                        if class_type == 'Linear':
                            # compute gradient of activation
                            if (not args.lora) or (args.lora and "base_layer" not in module.name): 
                                flops = module.flops
                                total_flops += flops
                                total_backward_flops += flops

                                # compute gradient of weight
                                if module.weight.requires_grad:
                                    total_flops += flops
                                    total_backward_flops += flops

                        elif class_type == 'LayerNorm': # TODO: norm and activation function just copy their forward, but this is not accurate!
                            flops = module.flops
                            total_flops += flops
                            total_backward_flops += flops

                        elif class_type == 'GELUActivation' or class_type == 'SiLUActivation':
                            flops = module.flops
                            total_flops += flops
                            total_backward_flops += flops

                        if 'query' in module.name:
                            # before: 2 (b, s, h) @ (b, h, s)
                            # after: 4 (b, s, h) @ (b, h, s) + 2 (b, s, s) @ (b, s, s)
                            attn_out_shape = module.attn_out_shape # (b, s, h)
                            b, s, h = attn_out_shape
                            flops = 2 * (4 * b * s * h * s + 2 * b * s * s * s)
                            total_flops += flops
                            total_backward_flops += flops
                    
                    # register forward hook
                    for name, module in model.named_modules():
                        module.name = name
                        module.register_forward_hook(flops_forward_hook)
                        module.register_backward_hook(flops_backward_hook)

                    out = model(**batch)
                    loss = out.logits.sum()
                    loss.backward()

                    print(f"Total FLOPs: {total_flops / 10**12} TFLOPs")
                    print(f"Total Forward FLOPs: {total_forward_flops / 10**12} TFLOPs")
                    print(f"Total Backward FLOPs: {total_backward_flops / 10**12} TFLOPs")

                    # macs, params = profile(model, inputs=inputs_for_flops,)

                    # print(f"Macs: {macs / 10**12} TFLOPs \t Params: {params / 10 ** 9} G")
                    # out_file = "get_macs.json"
                    # with open(out_file, 'w') as fout:
                    #     fout.write(json.dumps([macs, params]))
                    # print(f"save results to {out_file}")
                    exit()
                    
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps

                # log the train loss
                wandb.log({"train_loss": loss.item(), "train_step": train_step})

                if args.get_mem and iter > 1:
                    # accelerator.print("===============Before Backward=======================")
                    torch.cuda.synchronize()
                    before_backward = get_memory_usage(True)
                optimizer.zero_grad()
                loss.backward()
                torch.utils.checkpoint.first_iter = False
                if args.get_mem and iter > 1:
                    # accelerator.print("===============After Backward=======================")
                    torch.cuda.synchronize()
                    for t in batch:
                        del t
                    after_backward = get_memory_usage(False)  # model size
                    # init : weight + optimizer state + data size + grad (iter > 1)
                    # before backward : weight + optimizer state + data size + activation + loss + output + grad (iter > 1)
                    # after backward : init
                    # grad = weight
                    # total - act = weight + optimizer state + data size + loss + output + grad
                    total_mem.update(before_backward)
                    activation_mem.update(before_backward - after_backward)
                    peak_mem.update(
                        torch.cuda.max_memory_allocated()
                    )
                    del loss
                    del outputs
                    
                    accelerator.print("peak %d MB" % (peak_mem.get_value() / 1024 / 1024))
                    accelerator.print("total %d MB" % (total_mem.get_value() / 1024 / 1024))
                    accelerator.print("activation %d MB" % (activation_mem.get_value() / 1024 / 1024))
                    
                    accelerator.print("weight %d MB" % (get_weight_memory(model) / 1024 / 1024))
                    accelerator.print("gradient %d MB" % (get_gradient_memory(model) / 1024 / 1024))
                    accelerator.print("optimizer %d MB" % (get_optimizer_memory(optimizer) / 1024 / 1024))
                    accelerator.print('-------------------------------------------------')
                    exit(0)
                    
                if args.get_speed and iter > 1:
                    end.record()
                    torch.cuda.synchronize()
                    cur_batch_time = start.elapsed_time(end) / 1000.0 # event in ms
                    batch_total_time += cur_batch_time
                    
                    if iter == 6:
                        bs = args.per_device_train_batch_size
                        train_ips = 5 * bs / batch_total_time
                        res = "BatchSize: %d\tIPS: %.2f\t,Cost: %.2f ms" % (
                            bs, train_ips, 1000.0 / train_ips)
                        print(res, flush=True)
                        exp_recorder.record("network", args.model_name_or_path)
                        exp_recorder.record("algorithm", args.opt_level)
                        exp_recorder.record("batch_size", bs)
                        exp_recorder.record("ips", train_ips, 2)
                        exp_recorder.record("bacth_time", cur_batch_time)
                        exp_recorder.record("layer_num", config.num_hidden_layers)
                        exp_recorder.record("hidden_size", config.hidden_size)
                        exp_recorder.record("tstamp", time.time(), 2)
                        exp_recorder.dump('speed_results.json')
                        exit(0)

                # if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
                optimizer.step() #! optimizer generate there
                lr_scheduler.step()
                # optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break

                if args.gact:
                    def backprop():
                        small_batch = {}
                        for k, v in batch.items():
                            small_batch[k] = v[:8]
                        outputs = model(**small_batch)
                        loss = outputs.loss
                        optimizer.zero_grad()
                        loss.backward() #! gradient generate there
                        del loss
                        del outputs
                        del small_batch
                    controller.iterate(backprop)

            with torch.no_grad():
                model.eval()
                for step, batch in enumerate(eval_dataloader):
                    for k, v in batch.items():
                        batch[k] = v.to(args.device)
                    outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                    metric.add_batch(
                        predictions=accelerator.gather(predictions),
                        references=accelerator.gather(batch["labels"]),
                    )

            eval_metric = metric.compute()
            logger.info(f"epoch {epoch}: {eval_metric}")

            for key, value in eval_metric.items():
                wandb.log({f"val_{key}": value, "val_step": epoch})
            
            if eval_metric[metric_key[args.task_name]] > best_metric:
                best_metric = eval_metric[metric_key[args.task_name]]
                wandb.log({"best_val_acc": best_metric, "val_step": epoch})

            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    repo.push_to_hub(
                        commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                    )

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            with open(os.path.join(args.output_dir, 'result.txt'), 'a') as f:
                f.write('lr:%f, bsz:%d, result:%f\n' % (args.learning_rate, args.per_device_train_batch_size, best_metric))

        # if args.task_name == "mnli":
        #     # Final evaluation on mismatched validation set
        #     eval_dataset = processed_datasets["validation_mismatched"]
        #     eval_dataloader = DataLoader(
        #         eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        #     )
        #     eval_dataloader = accelerator.prepare(eval_dataloader)

        #     model.eval()
        #     for step, batch in enumerate(eval_dataloader):
        #         outputs = model(**batch)
        #         predictions = outputs.logits.argmax(dim=-1)
        #         metric.add_batch(
        #             predictions=accelerator.gather(predictions),
        #             references=accelerator.gather(batch["labels"]),
        #         )

        #     eval_metric = metric.compute()
        #     logger.info(f"mnli-mm: {eval_metric}")

    # eval_metric = metric.compute()
    # logger.info(f"epoch {epoch}: {eval_metric}")


if __name__ == "__main__":
    main()
