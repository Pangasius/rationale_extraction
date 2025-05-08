import random

import math
import os

from tqdm import tqdm
import wandb

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from nltk.tokenize import sent_tokenize
from datasets import load_metric
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
)

from processing.citation_dataset import compute_iou_mean
from baselines.baseline import print_metrics


class TextClassificationDataset(Dataset):
    def __init__(self, dataframe, oversample=False):
        self.df = dataframe
        if oversample:
            self.positive_example = self.df[self.df["label"] == 1]
            self.negative_example = self.df[self.df["label"] == 0]
            self.oversample()

    def __len__(self):
        return len(self.df)

    def oversample(self):
        ratio = len(self.negative_example) / len(self.positive_example)
        print("ratio before oversampling:", ratio)
        self.df = pd.concat(
            [
                self.negative_example,
                self.positive_example.sample(
                    n=len(self.negative_example), replace=True
                ),
            ]
        )
        print("Oversampling done")
        ratio = len(self.df[self.df["label"] == 1]) / len(
            self.df[self.df["label"] == 0]
        )
        print("ratio after oversampling:", ratio)

    def __getitem__(self, idx):
        label = self.df.iloc[idx]["label"]
        text = self.df.iloc[idx]["text"]
        input_ids = self.df.iloc[idx]["input_ids"]
        attention_mask = self.df.iloc[idx]["attention_mask"]

        return {
            "text": text,
            "label": torch.tensor(label),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def get_text(self, idx):
        return self.df.iloc[idx]["text"]

    def get_label(self, idx):
        return self.df.iloc[idx]["label"]


class CC_classifier:
    def __init__(
        self,
        model_name,
        dataset_input="A&Q",
        model_checkpoint=None,
        tokenizer_checkpoint=None,
    ):
        self.dataset_input = dataset_input
        self.model_name = model_name
        self.input_type = dataset_input
        if model_checkpoint is not None and tokenizer_checkpoint is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
            self.classifier = self.load_classifier(
                model_name, model_checkpoint, self.tokenizer
            )

    @staticmethod
    def load_classifier(model_name, model_checkpoint, tokenizer):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = None
        if model_name == "google/gemma-2b":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # enable 4-bit quantization
                bnb_4bit_quant_type="nf4",  # information theoretically optimal dtype for normally distributed weights
                bnb_4bit_use_double_quant=True,  # quantize quantized weights //insert xzibit meme
                bnb_4bit_compute_dtype=torch.bfloat16,  # optimized fp format for ML
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, quantization_config=quantization_config, num_labels=2
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        model = PeftModel.from_pretrained(
            model, model_checkpoint, adapter_name="lora"
        ).to(device)
        return model

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metric1 = load_metric("precision", trust_remote_code=True)
        metric2 = load_metric("recall", trust_remote_code=True)
        metric3 = load_metric("f1", trust_remote_code=True)
        metric4 = load_metric("accuracy", trust_remote_code=True)
        precision = metric1.compute(predictions=predictions, references=labels)[
            "precision"
        ]
        recall = metric2.compute(predictions=predictions, references=labels)["recall"]
        f1 = metric3.compute(predictions=predictions, references=labels)["f1"]
        accuracy = metric4.compute(predictions=predictions, references=labels)[
            "accuracy"
        ]
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    def make_class_dataframe(self, dataset, augment=0):
        text_array = []
        label_array = []
        input_ids_array = []
        attention_mask = []
        for ind in dataset.index:
            context = dataset["context"][ind]
            answer = dataset["answer"][ind]
            question = dataset["question"][ind]
            citation_text = dataset["citation"][ind]
            context_sentences = sent_tokenize(context)
            random_sentences = []
            for i in range(augment):
                random_context = dataset["context"][random.choice(dataset.index)]
                sentences = sent_tokenize(random_context)
                random_sentences.append(
                    sentences[random.randint(0, len(sentences) - 1)]
                )

            if self.input_type == "A":
                for i in range(len(context_sentences)):
                    text = (
                        "sentence to classify:\n"
                        + context_sentences[i]
                        + "\n\nAnswer:\n"
                        + answer
                    )
                    text_array.append(text)
                    embed = self.tokenizer(text, padding="max_length", truncation=True)
                    input_ids_array.append(embed["input_ids"])
                    attention_mask.append(embed["attention_mask"])

                    if context_sentences[i] in citation_text:
                        label_array.append(1)
                    else:
                        label_array.append(0)

                for i in range(len(random_sentences)):
                    text = (
                        "sentence to classify:\n"
                        + random_sentences[i]
                        + "\n\nAnswer:\n"
                        + answer
                    )
                    text_array.append(text)
                    embed = self.tokenizer(text, padding="max_length", truncation=True)
                    input_ids_array.append(embed["input_ids"])
                    attention_mask.append(embed["attention_mask"])
                    label_array.append(0)

            elif self.input_type == "A&Q":
                for i in range(len(context_sentences)):
                    text = (
                        "sentence to classify:\n"
                        + context_sentences[i]
                        + "\nAnswer:\n"
                        + answer
                        + "\nQuestion:\n"
                        + question
                    )
                    text_array.append(text)
                    embed = self.tokenizer.encode_plus(
                        text, padding="max_length", truncation=True, max_length=512
                    )
                    input_ids_array.append(embed["input_ids"])
                    attention_mask.append(embed["attention_mask"])

                    if context_sentences[i] in citation_text:
                        label_array.append(1)
                    else:
                        label_array.append(0)

                for i in range(len(random_sentences)):
                    text = (
                        "sentence to classify:\n"
                        + random_sentences[i]
                        + "\n\nAnswer:\n"
                        + answer
                        + "\nQuestion:\n"
                        + question
                    )
                    text_array.append(text)
                    embed = self.tokenizer.encode_plus(
                        text, padding="max_length", truncation=True, max_length=512
                    )
                    input_ids_array.append(embed["input_ids"])
                    attention_mask.append(embed["attention_mask"])
                    label_array.append(0)

            elif self.input_type == "S3&A":
                for i in range(len(context_sentences)):
                    if i == 0 and len(context_sentences) >= 2:
                        text = (
                            "sentence to classify:\n"
                            + "<!>"
                            + context_sentences[i]
                            + "<!>"
                            + context_sentences[i + 1]
                            + "\n\nAnswer:\n"
                            + answer
                        )
                    elif (
                        i == len(context_sentences) - 1 and len(context_sentences) >= 2
                    ):
                        text = (
                            "sentence to classify:\n"
                            + context_sentences[i - 1]
                            + "<!>"
                            + context_sentences[i]
                            + "<!>"
                            + "\n\nAnswer:\n"
                            + answer
                        )
                    elif len(context_sentences) >= 3:
                        text = (
                            "sentence to classify:\n"
                            + context_sentences[i - 1]
                            + "<!>"
                            + context_sentences[i]
                            + "<!>"
                            + context_sentences[i + 1]
                            + "\n\nAnswer:\n"
                            + answer
                        )
                    else:
                        text = (
                            "sentence to classify:\n"
                            + "<!>"
                            + context_sentences[i]
                            + "<!>"
                            + "\n\nAnswer:\n"
                            + answer
                        )

                    text_array.append(text)
                    embed = self.tokenizer.encode_plus(
                        text, padding="max_length", truncation=True, max_length=512
                    )
                    input_ids_array.append(embed["input_ids"])
                    attention_mask.append(embed["attention_mask"])

                    if context_sentences[i] in citation_text:
                        label_array.append(1)
                    else:
                        label_array.append(0)

            elif self.input_type == "S3&A&Q":
                for i in range(len(context_sentences)):
                    if i == 0 and len(context_sentences) >= 2:
                        text = (
                            "sentence to classify:\n"
                            + "<!>"
                            + context_sentences[i]
                            + "<!>"
                            + context_sentences[i + 1]
                            + "\n\nAnswer:\n"
                            + answer
                            + "\nQuestion:\n"
                            + question
                        )
                    elif (
                        i == len(context_sentences) - 1 and len(context_sentences) >= 2
                    ):
                        text = (
                            "sentence to classify:\n"
                            + context_sentences[i - 1]
                            + "<!>"
                            + context_sentences[i]
                            + "<!>"
                            + "\n\nAnswer:\n"
                            + answer
                            + "\nQuestion:\n"
                            + question
                        )
                    elif len(context_sentences) >= 3:
                        text = (
                            "sentence to classify:\n"
                            + context_sentences[i - 1]
                            + "<!>"
                            + context_sentences[i]
                            + "<!>"
                            + context_sentences[i + 1]
                            + "\n\nAnswer:\n"
                            + answer
                            + "\nQuestion:\n"
                            + question
                        )
                    else:
                        text = (
                            "sentence to classify:\n"
                            + "<!>"
                            + context_sentences[i]
                            + "<!>"
                            + "\n\nAnswer:\n"
                            + answer
                            + "\nQuestion:\n"
                            + question
                        )

                    text_array.append(text)
                    embed = self.tokenizer.encode_plus(
                        text, padding="max_length", truncation=True, max_length=512
                    )
                    input_ids_array.append(embed["input_ids"])
                    attention_mask.append(embed["attention_mask"])

                    if context_sentences[i] in citation_text:
                        label_array.append(1)
                    else:
                        label_array.append(0)

        class_df = pd.DataFrame(
            {
                "text": text_array,
                "label": label_array,
                "input_ids": input_ids_array,
                "attention_mask": attention_mask,
            }
        )
        return class_df

    def train(
        self,
        run_name,
        lr=5e-5,
        lora_drop=0.1,
        lora_r=1,
        augment=0,
        batch_size=8,
        train_size=1.0,
    ):
        access_token = os.getenv("ACCESS_TOKEN")

        lora_config = None

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # enable 4-bit quantization
            bnb_4bit_quant_type="nf4",  # information theoretically optimal dtype for normally distributed weights
            bnb_4bit_use_double_quant=True,  # quantize quantized weights //insert xzibit meme
            bnb_4bit_compute_dtype=torch.bfloat16,  # optimized fp format for ML
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = "[PAD]"

        if self.model_name == "google/gemma-2b":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                quantization_config=quantization_config,
                token=access_token,
                trust_remote_code=True,
            )
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                target_modules=["k_proj", "v_proj", "lm_head"],
                r=lora_r,
                lora_alpha=lora_r,
                lora_dropout=lora_drop,
            )

        if self.model_name == "mistralai/Mistral-7B-v0.1":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                quantization_config=quantization_config,
                token=access_token,
                trust_remote_code=True,
            )
            lora_config = LoraConfig(
                r=lora_r,  # the dimension of the low-rank matrices
                lora_alpha=lora_r,  # scaling factor for LoRA activations vs pre-trained weight activations
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,  # dropout probability of the LoRA layers
                bias="none",  # wether to train bias weights, set to 'none' for attention layers
                task_type="SEQ_CLS",
            )

        elif self.model_name == "distilbert-base-uncased":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                trust_remote_code=True,
            )
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                target_modules=["q_lin", "k_lin", "v_lin"],
                r=lora_r,
                lora_alpha=lora_r,
                lora_dropout=lora_drop,
            )

        elif self.model_name == "roberta-base":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                trust_remote_code=True,
            )
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_r,
                lora_dropout=lora_drop,
            )

        if not self.model:
            print("error model loading has failed")
            return

        self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)

        from processing.citation_dataset import CitationDataset

        dataset = CitationDataset("processing/annotated_datasets/clear_dataset.jsonl")

        splits = dataset.to_splits(shuffle=True, seed=42)

        train_set, val_set, _ = splits["train"].to_pandas(), splits["val"].to_pandas(), splits["test"].to_pandas()

        if train_size < 1.0:
            max_length = len(train_set)
            sep_index = math.floor(max_length * train_size)
            print("Training on ", sep_index, " samples")
            train_df = train_set[:sep_index]

        train_df = self.make_class_dataframe(train_set, augment=augment)
        val_df = self.make_class_dataframe(val_set)

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        train_dataset = TextClassificationDataset(train_df)
        eval_dataset = TextClassificationDataset(val_df)

        training_args = TrainingArguments(
            output_dir="llm_classifier/train_checkpoint/" + run_name,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=15,
            report_to="wandb",
            learning_rate=lr,
            run_name=run_name,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.model.save_pretrained("llm_classifier/train_model/model/" + run_name)
        self.tokenizer.save_pretrained("llm_classifier/train_model/tokenizer/" + run_name)
        wandb.finish()

    def make_prediction(self, answer, question, context, load_bar=None):
        context_sentences = sent_tokenize(context)
        pred_sentence = []
        pred_index = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.input_type == "A&Q":
            for i in range(len(context_sentences)):
                text = (
                    "sentence to classify:\n"
                    + context_sentences[i]
                    + "\nAnswer:\n"
                    + answer
                    + "\nQuestion:\n"
                    + question
                )
                inputs = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                score = self.classifier(**inputs)
                label = np.argmax(score.logits.detach().cpu().numpy(), axis=-1)[0]
                if label == 1:
                    pred_sentence.append(i)

        elif self.input_type == "S3&A":
            for i in range(len(context_sentences)):
                text = ""
                if i == 0 and len(context_sentences) >= 2:
                    text = (
                        "sentence to classify:\n"
                        + "<!>"
                        + context_sentences[i]
                        + "<!>"
                        + context_sentences[i + 1]
                        + "\n\nAnswer:\n"
                        + answer
                    )
                elif i == len(context_sentences) - 1 and len(context_sentences) >= 2:
                    text = (
                        "sentence to classify:\n"
                        + context_sentences[i - 1]
                        + "<!>"
                        + context_sentences[i]
                        + "<!>"
                        + "\n\nAnswer:\n"
                        + answer
                    )
                elif len(context_sentences) >= 3:
                    text = (
                        "sentence to classify:\n"
                        + context_sentences[i - 1]
                        + "<!>"
                        + context_sentences[i]
                        + "<!>"
                        + context_sentences[i + 1]
                        + "\n\nAnswer:\n"
                        + answer
                    )
                else:
                    text = (
                        "sentence to classify:\n"
                        + "<!>"
                        + context_sentences[i]
                        + "<!>"
                        + "\n\nAnswer:\n"
                        + answer
                    )

                inputs = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                score = self.classifier(**inputs)
                label = np.argmax(score.logits.detach().cpu().numpy(), axis=-1)[0]
                if label == 1:
                    pred_sentence.append(i)

        elif self.input_type == "S3&A&Q":
            for i in range(len(context_sentences)):
                text = ""
                if i == 0 and len(context_sentences) >= 2:
                    text = (
                        "sentence to classify:\n"
                        + "<!>"
                        + context_sentences[i]
                        + "<!>"
                        + context_sentences[i + 1]
                        + "\n\nAnswer:\n"
                        + answer
                        + "\nQuestion:\n"
                        + question
                    )
                elif i == len(context_sentences) - 1 and len(context_sentences) >= 2:
                    text = (
                        "sentence to classify:\n"
                        + context_sentences[i - 1]
                        + "<!>"
                        + context_sentences[i]
                        + "<!>"
                        + "\n\nAnswer:\n"
                        + answer
                        + "\nQuestion:\n"
                        + question
                    )
                elif len(context_sentences) >= 3:
                    text = (
                        "sentence to classify:\n"
                        + context_sentences[i - 1]
                        + "<!>"
                        + context_sentences[i]
                        + "<!>"
                        + context_sentences[i + 1]
                        + "\n\nAnswer:\n"
                        + answer
                        + "\nQuestion:\n"
                        + question
                    )
                else:
                    text = (
                        "sentence to classify:\n"
                        + "<!>"
                        + context_sentences[i]
                        + "<!>"
                        + "\n\nAnswer:\n"
                        + answer
                        + "\nQuestion:\n"
                        + question
                    )

                inputs = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                score = self.classifier(**inputs)
                label = np.argmax(score.logits.detach().cpu().numpy(), axis=-1)[0]
                if label == 1:
                    pred_sentence.append(i)

        accumulator = 0
        for i in range(len(context_sentences)):
            if i in pred_sentence:
                pred_index.append(
                    [accumulator, accumulator + len(context_sentences[i])]
                )
            accumulator += len(context_sentences[i]) + 1

        if load_bar:
            load_bar.update(1)
        return pred_index

    def compute_IoU(self, answer, question, context, citation, load_bar=None):
        pred = self.make_prediction(answer, question, context, load_bar)
        return compute_iou_mean([pred], [citation])

    def test(self, df):
        df = df.to_pandas()

        progress_bar = tqdm(total=df.shape[0], position=0, leave=True)
        df["prediction"] = df.apply(
            lambda row: self.make_prediction(
                row["answer"], row["question"], row["context"], progress_bar
            ),
            axis=1,
        )

        df["prediction"] = df.apply(lambda row: [row["prediction"]], axis=1)
        df["citation_index"] = df.apply(lambda row: [row["citation_index"]], axis=1)
        df["IoU"] = df.apply(
            lambda row: compute_iou_mean(row["prediction"], row["citation_index"]), axis=1
        )

        print_metrics(df, "Num_sentence", type="num_sentences")

        print("\n*************\nIoU:", df["IoU"].mean(), "\n*************")

        return df
