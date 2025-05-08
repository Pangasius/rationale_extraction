"""
Script to process the dataset from the Databricks Dolly 15k dataset.

First part splits the dataset in two parts for the two annotators.
Second part processes the annotations and gives a final dataset.
"""

import pandas as pd
import regex as re
from sympy import Interval, Union


def union(x):
    unified = Union(*x)
    return [unified] if isinstance(unified, Interval) else list(unified.args)


def dispatch_text(x):
    all_text = []
    for i in x:
        merged = {}
        for j in i:
            if j[1] in merged:
                merged[j[1]] += j[0]
            else:
                merged[j[1]] = j[0]
        all_text.append(list(merged.values()))
    return all_text


def merge_index(x):
    unified_interval = []
    for i in x:
        list_interval = None
        for inter in i:
            if not list_interval:
                list_interval = inter
            else:
                list_interval += inter

        if not list_interval:
            unified_interval.append(None)
        else:
            unified = Union(*list_interval)
            unified = [unified] if isinstance(unified, Interval) else list(unified.args)
            unified_interval.append(unified)
    return unified_interval


def dispatch_index(x):
    all_index = []
    for i in x:
        dispatch = {}
        for j in i:
            if j[2] not in dispatch:
                dispatch[j[2]] = []

            dispatch[j[2]].append(Interval.open(j[0], j[1]))
        dispatch_value = list(dispatch.values())
        for i in range(len(dispatch_value)):
            dispatch_value[i] = union(list(dispatch_value[i]))
        all_index.append(dispatch_value)

    return all_index


def process_sub(x):
    processed = []
    for i in x:
        line = []
        if i is None:
            processed.append(None)
            continue
        for j in i:
            if isinstance(j, Interval):
                j = [[int(j.start), int(j.end)]]
            else:
                j = [[int(k.start), int(k.end)] for k in j]
            line.append(j)
        processed.append(line)
    return processed


def process_citation(x):
    processed = []
    for i in x:
        if i is None:
            processed.append(None)
            continue
        if isinstance(i, Interval):
            i = [[int(i.start), int(i.end)]]
        else:
            i = [[int(k.start), int(k.end)] for k in i]
        processed.append(i)
    return processed


def merge_text(x):
    return [" ".join(i) for i in x]


if __name__ == "__main__":
    # Because we use doccano for annotation, we will merge
    # the instruction/context/response into one column
    separator = "\nCUSTOM_SEPARATOR\n"

    # Load datasets/databricks-dolly-15k/databricks-dolly-15k.jsonl
    # Make a strinfio object with the content of the file
    with open(
        "raw_datasets/databricks-dolly-15k/databricks-dolly-15k.jsonl", "r"
    ) as file:
        dataset = pd.read_json(
            path_or_buf=file, lines=True, orient="records", encoding="utf-8"
        )

    dataset = dataset[dataset["category"] == "closed_qa"]

    dataset_pair = dataset.iloc[::2]
    dataset_odd = dataset.iloc[1::2]

    dataset_1 = pd.DataFrame()
    dataset_1["text"] = (
        dataset_pair["instruction"]
        + separator
        + dataset_pair["context"]
        + separator
        + dataset_pair["response"]
    )
    dataset_1["label"] = ""

    dataset_2 = pd.DataFrame()
    dataset_2["text"] = (
        dataset_odd["instruction"]
        + separator
        + dataset_odd["context"]
        + separator
        + dataset_odd["response"]
    )
    dataset_2["label"] = ""

    dataset_1.to_json("parsed_datasets/dataset_1.jsonl", orient="records", lines=True)
    dataset_2.to_json("parsed_datasets/dataset_2.jsonl", orient="records", lines=True)

    # Make a strinfio object with the content of the file
    with open(
        "annotated_datasets/annotations_1.jsonl", "r", encoding="utf-8"
    ) as file:
        dfa = pd.read_json(
            path_or_buf=file, lines=True, orient="records", encoding="utf-8"
        )

    with open(
        "annotated_datasets/annotations_2.jsonl", "r", encoding="utf-8"
    ) as file:
        dfb = pd.read_json(
            path_or_buf=file, lines=True, orient="records", encoding="utf-8"
        )

    df = pd.concat([dfa, dfb]).reset_index(drop=True)
    df2 = df[
        df["Comments"].apply(
            lambda x: bool(re.search(r"'done'", str(x), re.IGNORECASE)) or x == []
        )
    ]

    # Create DataFrame with random data
    dff = df2.copy()

    # Get the index of the separators
    dff["Separator_Index"] = dff["text"].apply(
        lambda x: [m.start() for m in re.finditer(separator, x)]
    )
    # Split the text into ['Question', 'Context', 'Answer']
    dff[["question", "context", "answer"]] = dff["text"].str.split(
        separator, n=2, expand=True
    )

    # Drop unnecessary columns
    dff = dff.drop(columns=["text", "Comments"])

    # if label[0] and label[1] are before the separator[0]
    #   then the label is for the question
    # if label[0] and label[1] are after the separator[0]
    #   but before separator[1] then the label is for the context
    # if label[0] and label[1] are after the separator[1]
    #   then the label is for the answer

    # all label[2] that are the same refer to the same answer,
    # so they should be grouped together

    # data looks like this
    #   id	label	Separator_Index	Question	Context	Answer	Sub_answer	Citations
    # 0	888	[[0, 42, citation_1], [60, 164, citation_1], [...	[42, 570]	When did Virgin Australia start operating?	Virgin Australia, the trading name of Virgin A...	Virgin Australia commenced services on 31 Augu...

    # now create Sub_Question, Sub_Context and sub_answer

    sub_question = dff.apply(
        lambda x: [
            (x["question"][i[0]: i[1]], i[2])
            for i in x["label"]
            if i[0] < x["Separator_Index"][0] and i[1] <= x["Separator_Index"][0]
        ],
        axis=1,
    )
    sub_citation = dff.apply(
        lambda x: [
            (
                x["context"][
                    i[0] - x["Separator_Index"][0] - len(separator) : i[1]
                    - x["Separator_Index"][0]
                    - len(separator)
                ],
                i[2],
            )
            for i in x["label"]
            if i[0] > x["Separator_Index"][0] and i[1] <= x["Separator_Index"][1]
        ],
        axis=1,
    )
    sub_answer = dff.apply(
        lambda x: [
            (
                x["answer"][
                    i[0] - x["Separator_Index"][1] - len(separator) : i[1]
                    - x["Separator_Index"][1]
                    - len(separator)
                ],
                i[2],
            )
            for i in x["label"]
            if i[0] > x["Separator_Index"][1] and i[1] > x["Separator_Index"][1]
        ],
        axis=1,
    )

    sub_question_index = dff.apply(
        lambda x: [
            (i[0], i[1], i[2])
            for i in x["label"]
            if i[0] < x["Separator_Index"][0] and i[1] <= x["Separator_Index"][0]
        ],
        axis=1,
    )
    sub_citation_index = dff.apply(
        lambda x: [
            (
                i[0] - x["Separator_Index"][0] - len(separator),
                i[1] - x["Separator_Index"][0] - len(separator),
                i[2],
            )
            for i in x["label"]
            if i[0] > x["Separator_Index"][0] and i[1] <= x["Separator_Index"][1]
        ],
        axis=1,
    )
    sub_answer_index = dff.apply(
        lambda x: [
            (
                i[0] - x["Separator_Index"][1] - len(separator),
                i[1] - x["Separator_Index"][1] - len(separator),
                i[2],
            )
            for i in x["label"]
            if i[0] > x["Separator_Index"][1] and i[1] > x["Separator_Index"][1]
        ],
        axis=1,
    )

    # Merge the sub_question, sub_context and sub_answer
    dff["sub_question"] = dispatch_text(sub_question)
    dff["sub_citation"] = dispatch_text(sub_citation)
    dff["sub_answer"] = dispatch_text(sub_answer)

    dff["sub_question_index"] = process_sub(dispatch_index(sub_question_index))
    sub_citation_index_treated = dispatch_index(sub_citation_index)
    dff["sub_citation_index"] = process_sub(sub_citation_index_treated)
    dff["citation_index"] = process_citation(merge_index(sub_citation_index_treated))
    dff["citation"] = merge_text(dff["sub_citation"])
    dff["sub_answer_index"] = process_sub(dispatch_index(sub_answer_index))

    dff["num_sub_question"] = dff.apply(lambda row: len(row["sub_question"]), axis=1)

    dff = dff[dff["citation_index"].notnull()]

    # drop the label column and the Separator_Index column
    dff = dff.drop(columns=["label", "Separator_Index"])

    dff.to_json("annotated_datasets/clear_dataset.jsonl", orient="records", lines=True)

    print(dff)
