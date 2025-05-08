from typing import (
    Iterator,
)

import pandas as pd
from torch.utils.data import Dataset
from sympy import Union, Intersection, Interval

import datasets

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

SPECIAL_TOKENS = {
    "start_citation": "<start_citation_open>",
    "end_citation": "<end_citation_open>",
    "citation_close": "<citation_close>",
}


def compute_iou(pred, target):
    """
    pred and target are lists of open Interval object
    """

    # First cast every member of pred and target to set
    pred = [Interval.open(p[0], p[1]) for p in pred]
    target = [Interval.open(t[0], t[1]) for t in target]

    intersection = 0
    for p in pred:
        for t in target:
            intersection += float(Intersection(p, t).measure)

    union = float(Union(*pred, *target).measure)

    if union == 0:
        return 0

    return intersection / union


def compute_iou_list(pred_list, target_list):
    """
    We have a list of predictions and a list of targets
    A prediction or a target is a list of tuples (start, end)

    All target are always complete, but a prediction can be partial
    We will compute the IoU metric as the product of IoU score for
    each pair (pred, target)
    """

    # compute the IoU
    IoU_score = []
    for i in range(len(pred_list)):
        IoU_score.append(compute_iou(pred_list[i], target_list[i]))
    return IoU_score


def compute_iou_mean(prediction_indices, sub_citation_indices):
    """
    Computes the Intersection over Union (IoU) metric between each citation
    (prediction) and its corresponding context.

    Parameters:
        prediction_indices (list of list of tuples): Indices representing the
        start and end of citations in the prediction.
        sub_citation_indices (list of list of tuples): Indices representing
        the start and end of citations in the context.

    Returns:
        float: Mean IoU metric.

    """
    if not sub_citation_indices or not prediction_indices:
        return None

    iou_list = compute_iou_list(prediction_indices, sub_citation_indices)

    # Check for overlapping indices
    if any([iou > 1 for iou in iou_list]):
        print(
            "Either prediction or reference has overlapping indices.\
            Check the data.",
            prediction_indices,
            sub_citation_indices,
            iou_list,
        )
        return None

    return np.mean(iou_list)


def to_splits(df: pd.DataFrame, shuffle: bool, seed: int) -> datasets.DatasetDict:
    """Splits the dataset into three parts: train, test, and validation.

    Args:
        df (pd.DataFrame): The dataset to split.
        shuffle (bool): Whether to shuffle the dataset.
        seed (int): The seed for the random number generator.

    Returns:
        datasets.DatasetDict: The split dataset.
    """
    dataset = datasets.Dataset.from_dict(df.to_dict())

    train_test = dataset.train_test_split(test_size=0.1, shuffle=shuffle, seed=seed)

    train_test_val = train_test["train"].train_test_split(
        test_size=1 - 8 / 9, shuffle=shuffle, seed=seed
    )

    final = {
        "train": train_test_val["train"],
        "test": train_test["test"],
        "val": train_test_val["test"],
    }

    return datasets.DatasetDict(final)


class QADataset(Dataset):
    """
    Class to load and preprocess the QA dataset for attention-based methods.
    
    It can add special tokens around the citations in the answer if wanted and can be used to truncate the dataset to a specific length in tokens.
    """
    def __init__(self, root, no_special_tokens=False):
        with open(root, "r", encoding="utf-8") as file:
            self.dataframe = pd.read_json(
                path_or_buf=file, lines=True, orient="records", encoding="utf-8"
            )
        self.root = root

        self.special_tokens_dict = SPECIAL_TOKENS

        processed = zip(
            *self.dataframe.apply(
                lambda x: self.process(x, no_special_tokens=no_special_tokens), axis=1
            )
        )

        (
            self.dataframe["question"],
            self.dataframe["context"],
            self.dataframe["answer"],
        ) = processed

    def process(self, line, no_special_tokens=False):
        input_ = line["question"]
        context_ = line["context"]
        output_ = line["answer"]

        if no_special_tokens:
            return input_, context_, output_

        # we just need to add special tokens in the output
        # following "sub_answer_index"
        start_indices = []
        end_indices = []
        citation_number = 0
        for bound_array in line["sub_answer_index"]:
            citation_number += 1
            for bounds in bound_array:
                start_indices.append((citation_number, bounds[0]))
                end_indices.append((citation_number, bounds[1]))

        start_indices.sort(key=lambda x: x[1])
        end_indices.sort(key=lambda x: x[1])

        output_stream = ""
        start_index = 0
        end_index = 0
        i = 0
        while i < len(output_) + 1:
            if end_index < len(end_indices) and end_indices[end_index][1] == i:
                output_stream += (
                    self.special_tokens_dict["end_citation"]
                    + str(end_indices[end_index][0])
                    + self.special_tokens_dict["citation_close"]
                )
                end_index += 1
            elif (
                start_index < len(start_indices) and start_indices[start_index][1] == i
            ):
                output_stream += (
                    self.special_tokens_dict["start_citation"]
                    + str(start_indices[start_index][0])
                    + self.special_tokens_dict["citation_close"]
                )
                start_index += 1
            elif i < len(output_):
                output_stream += output_[i]
                i += 1
            else:
                i += 1

        return input_, context_, output_stream

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        line = self.dataframe.iloc[index]

        return line["question"], line["context"], line["answer"]

    def to_dict(self):
        dict = {"question": [], "context": [], "answer": [], "id": []}
        for i in range(len(self)):
            input_, context_, output_ = self[i]
            dict["question"].append(input_)
            dict["context"].append(context_)
            dict["answer"].append(output_)
            dict["id"].append(i)

        return dict

    def stats(self, tokenizer, truncate=None, save=False):
        datasets_dataset = datasets.Dataset.from_dict(self.to_dict())

        # Create a new dataframe with the 'question', 'context', 'answer' columns
        df = datasets_dataset.to_pandas()

        if isinstance(df, Iterator):
            df = pd.concat(df, ignore_index=True)

        # Create a new column with the length of the 'question', 'context', 'answer'
        df["question_len"] = df["question"].apply(len)
        df["context_len"] = df["context"].apply(len)
        df["answer_len"] = df["answer"].apply(len)

        # Find out the token length of the 'question', 'context', 'answer'
        df["question_token_len"] = df["question"].apply(tokenizer.tokenize).apply(len)
        df["context_token_len"] = df["context"].apply(tokenizer.tokenize).apply(len)
        df["answer_token_len"] = df["answer"].apply(tokenizer.tokenize).apply(len)

        if truncate is not None:
            df = df[(df["question_token_len"] + df["context_token_len"]) <= truncate[0]]
            df = df[df["answer_token_len"] <= truncate[1] - truncate[0]]

        def plot_distribution(x, x_token, title):
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            sns.histplot(x, bins=50)
            plt.title(f"{title} length")
            plt.subplot(1, 2, 2)
            sns.histplot(x_token, bins=50)
            plt.title(f"{title} token length")

            if save:
                plt.savefig(f"{title}_distribution.png")
            else:
                plt.show()

        plot_distribution(df["question_len"], df["question_token_len"], "question")
        plot_distribution(df["context_len"], df["context_token_len"], "context")
        plot_distribution(df["answer_len"], df["answer_token_len"], "answer")

        # Find out the 95th percentile of the token length of the 'question', 'context', 'answer'
        question_token_len_95 = df["question_token_len"].quantile(0.95)
        context_token_len_95 = df["context_token_len"].quantile(0.95)
        answer_token_len_95 = df["answer_token_len"].quantile(0.95)

        print(
            f"95th percentile of the token length of the input (question + context): {question_token_len_95 + context_token_len_95} (question: {question_token_len_95}, context: {context_token_len_95})"
        )
        print(
            f"95th percentile of the token length of the answer: {answer_token_len_95}"
        )

    @staticmethod
    def truncate(data: datasets.DatasetDict, tokenizer, truncate):
        """Truncates the dataset based on the token length of the 'question', 'context', 'answer'.
        The truncation is done based on the sum of the token length of the 'question' and 'context' (truncate[0]) and the token length of the 'answer' (truncate[1] - truncate[0]).

        Args:
            data (datasets.DatasetDict): The dataset to truncate.
            tokenizer (_type_): The tokenizer to use.
            truncate (_type_): The truncation lengths.

        Returns:
            datasets.DatasetDict: The truncated dataset.
        """
        for key, dataset in data.items():
            # Create a new dataframe with the 'question', 'context', 'answer' columns
            df = dataset.to_pandas()

            if isinstance(df, Iterator):
                df = pd.concat(df, ignore_index=True)

            df["question_token_len"] = (
                df["question"].apply(tokenizer.tokenize).apply(len)
            )
            df["context_token_len"] = df["context"].apply(tokenizer.tokenize).apply(len)
            df["answer_token_len"] = df["answer"].apply(tokenizer.tokenize).apply(len)

            df = df[(df["question_token_len"] + df["context_token_len"]) <= truncate[0]]
            df = df[df["answer_token_len"] <= truncate[1] - truncate[0]]

            df = df.drop(
                columns=["question_token_len", "context_token_len", "answer_token_len"]
            )

            data[key] = datasets.Dataset.from_pandas(df)

        return data

    def to_splits(self, shuffle: bool, seed: int):
        return to_splits(self, shuffle, seed)


class CitationDataset(Dataset):
    """
    Convenient class to load the citation dataset and transform it to a dictionary.
    """
    def __init__(self, root):
        with open(root, "r", encoding="utf-8") as file:
            self.dataframe = pd.read_json(
                path_or_buf=file, lines=True, orient="records", encoding="utf-8"
            )
        self.root = root

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        return self.dataframe.iloc[index]

    def to_dict(self):
        dict = {
            "question": [],
            "context": [],
            "answer": [],
            "id": [],
            "sub_question": [],
            "sub_citation": [],
            "sub_answer": [],
            "sub_question_index": [],
            "sub_citation_index": [],
            "sub_answer_index": [],
            "citation_index": [],
            "citation": [],
            "num_sub_question": [],
        }

        for i in range(len(self)):
            dict["question"].append(self.dataframe.iloc[i]["question"])
            dict["context"].append(self.dataframe.iloc[i]["context"])
            dict["answer"].append(self.dataframe.iloc[i]["answer"])
            dict["id"].append(i)
            dict["sub_question"].append(self.dataframe.iloc[i]["sub_question"])
            dict["sub_citation"].append(self.dataframe.iloc[i]["sub_citation"])
            dict["sub_answer"].append(self.dataframe.iloc[i]["sub_answer"])
            dict["sub_question_index"].append(
                self.dataframe.iloc[i]["sub_question_index"]
            )
            dict["sub_citation_index"].append(
                self.dataframe.iloc[i]["sub_citation_index"]
            )
            dict["sub_answer_index"].append(self.dataframe.iloc[i]["sub_answer_index"])
            dict["citation_index"].append(self.dataframe.iloc[i]["citation_index"])
            dict["citation"].append(self.dataframe.iloc[i]["citation"])
            dict["num_sub_question"].append(len(self.dataframe.iloc[i]["sub_question"]))

        return dict

    def to_splits(self, shuffle: bool, seed: int):
        return to_splits(self, shuffle, seed)
