"""
Convenience script to run different parts of the project.

Important arguments can be found in the README.md file.
"""

import os

import argparse

import rich

from dotenv import load_dotenv

from huggingface_hub import login

import wandb

import pandas as pd
import numpy as np
import nltk

import math

from tqdm import tqdm


def rl_training(parser):
    parser.add_argument("--base_path", type=str, default="attention/models/results")
    parser.add_argument("--model_name", type=str, default="google/gemma-2b")
    parser.add_argument("--from_checkpoint", type=str, default="latest")
    parser.add_argument("--loading_checkpoint", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--head", type=int, default=5)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--reward", type=str, default="attn")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=256 + 64)
    parser.add_argument("--max_length_input", type=int, default=256)
    parser.add_argument("--train_length", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_first", default=False, action="store_true")
    args = parser.parse_args()

    rich.print("\n\n")
    rich.print("Running with custom args: ", vars(args))
    rich.print("\n\n")

    reward_values = [
        "attn",
        "meteor",
        "meteor+attn",
        "meteor+attn-pen",
        "meteor+attn-fancy",
        "NG",
        "meteor+NG-fancy",
        "meteor+classify-fancy",
    ]
    if args.reward not in reward_values:
        raise ValueError(f"Reward must be in {reward_values}, got {args.reward}.")

    from attention.learning_tokens_utils import (
        preprocess_function,
    )

    from processing.citation_dataset import QADataset, CitationDataset

    import evaluate

    from attention.rl_training_utils import (
        get_rl_model,
        get_rl_trainer,
        rl_train,
    )

    if args.from_checkpoint == "latest":
        files = os.listdir("attention/models/results/" + args.model_name)

        if len(os.listdir("attention/models/results/" + args.model_name)) == 0:
            raise ValueError("No checkpoint found in the directory")

        if any("." in file for file in files):
            args.from_checkpoint = ""
        else:
            # List all directory directly under attention/models/results
            args.from_checkpoint = sorted(
                files, key=lambda x: int(x.split("/")[-1].split("-")[-1])
            )[-1]

    load_dotenv()
    read_token = os.getenv("READ_TOKEN")

    rich.print("Loading model")

    tokenizer, model = get_rl_model(
        args.model_name,
        access_token=read_token,
        from_checkpoint=args.from_checkpoint,
        save_model=args.save_first,
        version=args.version,
    )

    rich.print("Loading datasets")

    dataset = QADataset(
        "processing/annotated_datasets/clear_dataset.jsonl", no_special_tokens=True
    )
    c_dataset = CitationDataset("processing/annotated_datasets/clear_dataset.jsonl")

    splits = dataset.to_splits(shuffle=True, seed=args.seed)

    rich.print("Sizes of splits before", {k: len(v) for k, v in splits.items()})

    splits = QADataset.truncate(
        splits, tokenizer, (args.max_length_input, args.max_length)
    )

    rich.print("Sizes of splits after", {k: len(v) for k, v in splits.items()})

    tokenized_splits_clm = splits.map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            (args.max_length_input, args.max_length),
            out_labels=True,
            padding=False,
        ),
        batched=True,
        remove_columns=["answer", "context", "question"],
    )

    rich.print("Loading trainer")

    wandb_args = {
        "wandb": {
            "config": vars(args),
        }
    }

    trainer = get_rl_trainer(
        model,
        tokenizer,
        tokenized_splits_clm["train"],
        args.seed,
        model_name=args.model_name,
        batch_size=args.batch_size,
        tracker_args=wandb_args,
    )

    rich.print("Loading metrics")

    metrics = [
        evaluate.load("meteor"),
        evaluate.load("bleu"),
        evaluate.load("rouge"),
        # evaluate.load("bertscore", lang="en"),
    ]

    rich.print("Starting training")

    evaluate_dataloader = get_rl_trainer(
        model,
        tokenizer,
        tokenized_splits_clm["val"],
        args.seed,
        model_name=args.model_name,
        tracker_args=wandb_args,
    ).dataloader

    rl_train(
        trainer=trainer,
        tokenizer=tokenizer,
        truncate=(args.max_length_input, args.max_length),
        metrics=metrics,
        layer=args.layer,
        head=args.head,
        reward_type=args.reward,
        evaluate_dataloader=evaluate_dataloader,
        epoch=args.train_length,
        rank=args.rank,
        threshold=args.threshold,
        c_dataset=c_dataset,
    )

    path = (
        "attention/models/trl/results/"
        + args.model_name
        + "/trained_"
        + args.from_checkpoint
        + "_"
        + str(args.reward)
        + "_L"
        + str(args.layer)
        + "_H"
        + str(args.head)
        + "_S"
        + str(args.seed)
        + "_M"
        + str(args.max_length)
        + "_MI"
        + str(args.max_length_input)
        + "_V"
        + str(args.version)
        + "_T"
        + str(args.train_length)
    )

    index = 1

    while os.path.exists(path + "_" + str(index)):
        index += 1

    rich.print("Saving model to", path + "_" + str(index))

    trainer.save_pretrained(path + "_" + str(index))


def learning_tokens(parser):
    parser.add_argument("--base_path", type=str, default="attention/models/results")
    parser.add_argument("--model_name", type=str, default="google/gemma-2b")
    parser.add_argument("--from_checkpoint", type=str, default="latest")
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=1300)
    parser.add_argument("--max_length_input", type=int, default=1100)
    parser.add_argument("--save_attention", default=False, action="store_true")
    parser.add_argument("--no_special_tokens", default=False, action="store_true")
    args = parser.parse_args()

    rich.print("\n\n")
    rich.print("Running with custom args: ", vars(args))
    rich.print("\n\n")

    loading_checkpoint = args.from_checkpoint != "none"

    if args.from_checkpoint == "latest":
        files = os.listdir("attention/models/results/" + args.model_name)

        if len(os.listdir("attention/models/results/" + args.model_name)) == 0:
            raise ValueError("No checkpoint found in the directory")

        if any("." in file for file in files):
            args.from_checkpoint = ""
        else:
            # List all directory directly under attention/models/results
            args.from_checkpoint = sorted(
                files, key=lambda x: int(x.split("/")[-1].split("-")[-1])
            )[-1]

    # Make a nice description in text with rich

    train_text = "train" if args.train else "inference"

    # Imported here to allow for CUDA_VISIBLE_DEVICES to be set
    from processing.citation_dataset import QADataset

    from attention.learning_tokens_utils import (
        preprocess_function,
        get_trainer,
        inference,
        get_prepared_model,
    )

    base_lora_text = (
        "Base" if args.from_checkpoint == "none" and not args.train else "LoRA"
    )
    no_spec = "NO_SPEC " if args.no_special_tokens else ""

    # Start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="CC-new-tokens",
        save_code=True,
        name=f"{no_spec}{base_lora_text} {args.model_name}/{args.from_checkpoint} {train_text}",
        config=vars(args),
    )

    # Huggingface token
    load_dotenv()
    read_token = os.getenv("READ_TOKEN")
    write_token = os.getenv("WRITE_TOKEN")

    if read_token is None:
        raise ValueError("No read token found in .env")
    if write_token is None:
        raise ValueError("No write token found in .env")

    login(token=read_token)

    # Loading
    dataset = QADataset(
        "processing/annotated_datasets/clear_dataset.jsonl",
        no_special_tokens=args.no_special_tokens,
    )

    tokenizer, model = get_prepared_model(
        model_name=args.model_name,
        from_checkpoint=args.from_checkpoint,
        access_token=read_token,
        train=args.train,
        save_attention=args.save_attention,
        loading_checkpoint=loading_checkpoint,
        no_special_tokens=args.no_special_tokens,
    )

    splits = dataset.to_splits(shuffle=True, seed=args.seed)

    # Reduce dataset size for testing
    # for split in splits:
    #    splits[split] = splits[split].select(range(5))

    splits = QADataset.truncate(
        splits, tokenizer, (args.max_length_input, args.max_length)
    )

    splits = splits.select_columns(["id", "answer", "context", "question"])

    if args.train:
        tokenized_splits_clm = splits.map(
            lambda x: preprocess_function(
                x, tokenizer, (args.max_length_input, args.max_length), out_labels=False
            ),
            batched=True,
            remove_columns=["id", "answer", "context", "question"],
        )

        trainer = get_trainer(
            model_name=args.model_name,
            model=model,
            train_dataset=tokenized_splits_clm["train"],
            eval_dataset=tokenized_splits_clm["val"],
            tokenizer=tokenizer,
            seed=args.seed,
            access_token=write_token,
            no_special_tokens=args.no_special_tokens,
        )

        trainer.train()

    else:
        tokenized_splits_inf = splits.map(
            lambda x: preprocess_function(
                x, tokenizer, (args.max_length_input, args.max_length), out_labels=True
            ),
            batched=True,
        )

        for on_dataset in ["val"]:  # , "train"]:
            df = inference(
                model,
                tokenized_splits_inf,
                tokenizer,
                (args.max_length_input, args.max_length),
                on_dataset,
                1,
                args.save_attention,
            )

            wandb.log({f"table {on_dataset}": wandb.Table(dataframe=df)})

    wandb.finish()


def analyse_attention_results(args):
    parser.add_argument("--iou_path", type=str, required=True)
    args = parser.parse_args()

    rich.print("\n\n")
    rich.print("Running with custom args: ", vars(args))
    rich.print("\n\n")

    # Read scores
    with open(args.iou_path, "r") as f:
        df = pd.read_csv(f)

    # match name of columns
    df.rename(columns={"dataset_id": "id"}, inplace=True)
    df.rename(columns={"iou": "IoU"}, inplace=True)

    from processing.citation_dataset import QADataset

    # Read ref sentences
    c_dataset = QADataset("processing/annotated_datasets/clear_dataset.jsonl", no_special_tokens=True)

    c_dataset = pd.DataFrame.from_dict(c_dataset.to_dict())

    # merge scores and ref sentences
    merged = df.merge(c_dataset[["id", "context"]], on="id")

    merged['num_sentences'] = merged['context'].apply(lambda x: len(nltk.sent_tokenize(x)))

    # compute best scores
    iou_mean = df.groupby(["layer", "head", "rank", "threshold"])["IoU"].mean()

    df_grouped = pd.DataFrame({"iou_mean": iou_mean})
    df_grouped = df_grouped.sort_values("iou_mean", ascending=False)

    index_of_max = df_grouped["iou_mean"].idxmax()

    print(df_grouped)

    from baselines.baseline import print_metrics

    # print per sentence
    layer, head, rank, threshold = index_of_max
    merged_correct = merged[(merged["layer"] == layer) & (merged["head"] == head) & (merged["rank"] == rank) & (merged["threshold"] == threshold)]

    print_metrics(merged_correct, "", "num_sentences")


def analyse_attention(args):
    parser.add_argument("--model_name", type=str, default="google/gemma-2b")
    parser.add_argument("--trl", default=False, action="store_true")
    parser.add_argument("--from_checkpoint", type=str, default="latest")
    parser.add_argument("--max_length", type=int, default=1300)
    parser.add_argument("--max_length_input", type=int, default=1100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 3, 4, math.inf])
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0005, 0.0003, 0.0002, 0.0001, 0.0],
    )
    parser.add_argument("--heads", nargs="+", type=int, default=list(range(8)))
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(range(18)),
    )
    parser.add_argument("--no_special_tokens", default=False, action="store_true")
    args = parser.parse_args()

    rich.print("\n\n")
    rich.print("Running with custom args: ", vars(args))
    rich.print("\n\n")

    if args.from_checkpoint == "latest":
        files = os.listdir("attention/models/results/" + args.model_name)

        if len(os.listdir("attention/models/results/" + args.model_name)) == 0:
            raise ValueError("No checkpoint found in the directory")

        if any("." in file for file in files):
            args.from_checkpoint = ""
        else:
            # List all directory directly under attention/models/results
            args.from_checkpoint = sorted(
                files, key=lambda x: int(x.split("/")[-1].split("-")[-1])
            )[-1]

    # Huggingface token
    load_dotenv()
    read_token = os.getenv("READ_TOKEN")

    if args.split not in ["train", "val", "test"]:
        raise ValueError("Split must be one of train, val or test")

    if read_token is None:
        raise ValueError("No read token found in .env")

    # Prepare the path
    path = "attention/models/results/" + args.model_name + "/attention"

    # Create folder if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    name = (
        args.model_name.split("/")[-1]
        + "-CKP"
        + args.from_checkpoint
        + "-"
        + args.split
        + "-S"
        + str(args.seed)
        + ("-TRL" if args.trl else "")
        + ".csv"
    )

    # If file exists, raise error
    if os.path.exists(path + "/answer-" + name) or os.path.exists(
        path + "/iou-" + name
    ):
        raise ValueError("File already exists: " + path + "/" + name)

    rich.print("File will be saved at: " + path + "/" + name)

    # Imported here to allow for CUDA_VISIBLE_DEVICES to be set
    from attention.analyse_attention_utils import analyse_attention
    from attention.learning_tokens_utils import get_prepared_model, preprocess_function
    from processing.citation_dataset import QADataset, CitationDataset
    from transformers import DataCollatorForLanguageModeling
    from torch.utils.data import DataLoader
    from peft import PeftModel
    import torch

    # Loading
    dataset = QADataset(
        "processing/annotated_datasets/clear_dataset.jsonl",
        no_special_tokens=args.no_special_tokens,
    )
    c_dataset = CitationDataset("processing/annotated_datasets/clear_dataset.jsonl")

    splits = dataset.to_splits(shuffle=True, seed=args.seed)

    tokenizer, model = get_prepared_model(
        model_name=args.model_name,
        from_checkpoint=args.from_checkpoint,
        access_token=read_token,
        save_attention=True,
        no_special_tokens=args.no_special_tokens,
        loading_checkpoint=args.from_checkpoint != "none",
        trl=args.trl,
        device_map="auto",
    )

    if isinstance(model, PeftModel):
        model = model.merge_and_unload()

    splits = QADataset.truncate(
        splits, tokenizer, (args.max_length_input, args.max_length)
    )

    splits = splits.select_columns(["id", "answer", "context", "question"])

    tokenized_splits = splits.map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            (args.max_length_input, args.max_length),
            out_labels=True,
            padding=False,
        ),
        batched=True,
        batch_size=1,
        remove_columns=["answer", "context", "question"],
    )

    sel_ds = tokenized_splits[args.split].select_columns(
        ["id", "input_ids", "labels", "attention_mask"]
    )

    def collator(data):
        data_dict = {}

        for key in data[0]:
            if "input_ids" in key:
                data_dict[key] = [torch.Tensor(d[key]).to(torch.int32) for d in data]
            elif "labels" in key:
                data_dict[key] = [torch.Tensor(d[key]).to(torch.int32) for d in data]
            elif "attention_mask" in key:
                data_dict[key] = [torch.Tensor(d[key]).to(torch.bool) for d in data]
            else:
                data_dict[key] = [d[key] for d in data]

        return data_dict

    dataloader = DataLoader(
        sel_ds,
        batch_size=1,
        collate_fn=collator,
        pin_memory=True,
    )

    # Start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="CC-attention",
        save_code=True,
        name=name,
        config=vars(args),
    )

    iou_df = analyse_attention(
        model=model,
        tokenizer=tokenizer,
        path=path,
        name=name,
        layers=args.layers,
        heads=args.heads,
        ranks=args.ranks,
        thresholds=args.thresholds,
        truncate=(args.max_length_input, args.max_length),
        dataloader=dataloader,
        c_dataset=c_dataset,
    )

    wandb.log({f"table iou {args.split}": wandb.Table(dataframe=iou_df)}, commit=False)

    wandb.finish()


def llm_classifier(parser):
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora_drop", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--dataset_input", type=str, default="A&Q")
    parser.add_argument("--augment", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_size", type=float, default=1.0)
    args = parser.parse_args()

    rich.print("\n\n")
    rich.print("Running with custom args: ", vars(args))
    rich.print("\n\n")

    tqdm.pandas()

    load_dotenv()
    entity = os.getenv("WANDB_ENTITY")

    os.environ["WANDB_PROJECT"] = "CC-text-Classifier"
    os.environ["WANDB_ENTITY"] = entity
    wandb.login()

    from llm_classifier.llm_classifier import CC_classifier

    if args.run_name == "":
        args.run_name += args.model_name + "_"
        args.run_name += args.dataset_input + "_"
        args.run_name += str(args.augment) + "_"
        args.run_name += "r" + str(args.lora_r) + "_"
        args.run_name += "drop" + str(args.lora_drop) + "_"
        args.run_name += "+" + str(args.augment) + "_"
        args.run_name += "size" + str(args.train_size)

    if args.train:
        classifier = CC_classifier(args.model_name, args.dataset_input)
        classifier.train(
            args.run_name,
            args.lr,
            args.lora_drop,
            args.lora_r,
            args.augment,
            args.batch_size,
            args.train_size,
        )

    if args.test:
        classifier = CC_classifier(
            args.model_name,
            args.dataset_input,
            "llm_classifier/train_model/model/" + args.run_name,
            "llm_classifier/train_model/tokenizer/" + args.run_name,
        )
        from processing.citation_dataset import CitationDataset

        dataset = CitationDataset("processing/annotated_datasets/clear_dataset.jsonl")

        splits = dataset.to_splits(shuffle=True, seed=42)
        classifier.test(splits["test"])


def evaluate_classifier(parser):
    run_names = [
        "roberta-base_S3&A&Q_0_r4_drop0.1_+0_size1.0",
        "roberta-base_S3&A&Q_0_r4_drop0.1_+0_size1.0",
    ]

    from llm_classifier.llm_classifier import CC_classifier

    for run_name in run_names:
        classifier = CC_classifier(
            "roberta-base",
            "S3&A&Q",
            "llm_classifier/train_model/model/" + run_name,
            "llm_classifier/train_model/tokenizer/" + run_name,
        )
        from processing.citation_dataset import CitationDataset

        dataset = CitationDataset("processing/annotated_datasets/clear_dataset.jsonl")

        splits = dataset.to_splits(shuffle=True, seed=42)
        classifier.test(splits["test"])


def baseline(parser):
    parser.add_argument("--ft", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--method", type=str, default="ngramsTopK")
    parser.add_argument("--model_name", type=str, default="bert")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--ft_step", type=int, default=90)
    parser.add_argument("--ft_start", type=float, default=0.1)
    parser.add_argument("--ft_end", type=float, default=0.9)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--plot", default=False, action="store_true")
    parser.add_argument("--plot_dist_dataset", default=False, action="store_true")
    args = parser.parse_args()

    rich.print("\n\n")
    rich.print("Running with custom args: ", vars(args))
    rich.print("\n\n")

    from baselines.baseline import (
        dataset_distribution,
        ensemble_distribution,
        fine_tune_threshold,
        evaluate_TopK_embedding,
        evaluate_TopK_NGrams,
        evaluate_threshold,
    )

    from processing.citation_dataset import CitationDataset

    dataset = CitationDataset("processing/annotated_datasets/clear_dataset.jsonl")

    splits = dataset.to_splits(shuffle=True, seed=42)

    train_set, val_set, test_set = (
        splits["train"].to_pandas(),
        splits["val"].to_pandas(),
        splits["test"].to_pandas(),
    )

    if args.plot_dist_dataset:
        dataset_distribution(
            test_set, "number_sentences", "Number of sentences distribution eval"
        )
        dataset_distribution(
            val_set, "number_sentences", "Number of sentences distribution val"
        )
        dataset_distribution(
            train_set, "number_sentences", "Number of sentences distribution train"
        )
        ensemble_distribution(train_set, val_set, test_set)

    if args.ft:
        print(args.method)
        if args.method != "embeddingThreshold" and args.method != "ngramsThreshold":
            raise ValueError("Method not recognized")

        model = ""
        if args.method == "embeddingThreshold":
            model = args.model_name
        elif args.method == "ngramsThreshold":
            model = "ngrams"
        threshold_space = np.linspace(args.ft_start, args.ft_end, args.ft_step)
        fine_tune_threshold(dataset, model, threshold_space, batch_size=args.batch_size)

    if args.test:
        if args.method == "ngramsTopK":
            evaluate_TopK_NGrams(test_set, args.k, args.plot)
        elif args.method == "embeddingTopK":
            evaluate_TopK_embedding(
                test_set, args.model_name, args.k, args.plot, args.batch_size
            )
        elif args.method == "embeddingThreshold":
            if not args.threshold:
                raise ValueError("Threshold must be set")
            evaluate_threshold(
                test_set, args.threshold, args.model_name, args.batch_size
            )
        elif args.method == "ngramsThreshold":
            evaluate_threshold(test_set, args.threshold, "ngrams")
        else:
            raise ValueError("Method not recognized")


if __name__ == "__main__":
    nltk.download("punkt")

    # Params
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, required=True)
    parser.add_argument("--gpu", nargs="+", type=int, default=[0])
    parser.add_argument("--wandb", default=False, action="store_true")
    args_parsed, unknown = parser.parse_known_args()

    rich.print("\n\n")
    rich.print("Running with args: ", vars(args_parsed))
    rich.print("\n\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args_parsed.gpu))

    os.environ["WANDB__SERVICE_WAIT"] = "300"

    if not args_parsed.wandb:
        os.environ["WANDB_MODE"] = "dryrun"

    if args_parsed.script == "rl_training":
        rl_training(parser)
    elif args_parsed.script == "learning_tokens":
        learning_tokens(parser)
    elif args_parsed.script == "analyse_attention":
        analyse_attention(parser)
    elif args_parsed.script == "analyse_attention_results":
        analyse_attention_results(parser)
    elif args_parsed.script == "llm_classifier":
        llm_classifier(parser)
    elif args_parsed.script == "evaluate_classifier":
        evaluate_classifier(parser)
    elif args_parsed.script == "baseline":
        baseline(parser)
    else:
        raise ValueError("Script not recognized")
