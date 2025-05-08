import wandb
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv


def process_metric(df, metric_name):
    metric = {}
    for index, row in df.iterrows():
        epoch = math.ceil(row["train/epoch"])
        if epoch not in metric.keys():
            metric[epoch] = []
        if not np.isnan(row[metric_name]):
            metric[epoch].append(row[metric_name])
    for key in metric.keys():
        metric[key] = np.mean(metric[key])

    return metric.keys(), list(metric.values())


def plot_training(metric, run_names, legends, type="min-max", title="loss"):
    api = wandb.Api()

    load_dotenv()
    entity = os.getenv("WANDB_ENTITY")
    project = "CC-text-Classifier"

    runs = api.runs(entity + "/" + project)
    new_legends = ["" for _ in range(len(run_names))]
    values = [[] for _ in range(len(run_names))]
    refine_value = [{} for _ in range(len(run_names))]
    for run in runs:
        for i, name in enumerate(run_names):
            if run.name == name:
                new_legends[i] = run.name

                df = run.history()
                epochs, metric_values = process_metric(df, metric)
                if len(epochs) >= 15:
                    values[i].append(metric_values[:15])

    for i in range(len(refine_value)):
        key = "mean"
        refine_value[i][key] = np.mean(values[i], axis=0)
        key = "var"
        refine_value[i][key] = np.var(values[i], axis=0)
        key = "max"
        refine_value[i][key] = np.max(values[i], axis=0)
        key = "min"
        refine_value[i][key] = np.min(values[i], axis=0)
        key = "cnt"
        refine_value[i][key] = np.sum(values[i], axis=0)

    plt.figure()
    fig, ax = plt.subplots()
    for i in range(len(refine_value)):
        x = range(1, 16)
        y = refine_value[i]["mean"]
        plt.plot(x, y, label=legends[i])
        if type == "min-max":
            y_low = refine_value[i]["min"]
            y_high = refine_value[i]["max"]
            ax.fill_between(x, y_low, y_high, alpha=0.5, linewidth=0)
        elif type == "var":
            y_low = y - np.sqrt(refine_value[i]["var"])
            y_high = y + np.sqrt(refine_value[i]["var"])
            ax.fill_between(x, y_low, y_high, alpha=0.5, linewidth=0)
        elif type == "c95":
            z = 1.96
            y_low = y - z * (
                np.sqrt(refine_value[i]["var"]) / np.sqrt(refine_value[i]["cnt"])
            )
            y_high = y + z * (
                np.sqrt(refine_value[i]["var"]) / np.sqrt(refine_value[i]["cnt"])
            )
            ax.fill_between(x, y_low, y_high, alpha=0.5, linewidth=0)

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    plt.savefig("baselines/plots/{}.png".format(title))


def plot_metrics():
    run_names = [
        "distilbert-base-uncased_A&Q_0_r4_drop0.1_+0_size1.0",
        "distilbert-base-uncased_S3&A&Q_0_r4_drop0.1_+0_size1.0",
        "roberta-base_A&Q_0_r4_drop0.1_+0_size1.0",
        "roberta-base_S3&A&Q_0_r4_drop0.1_+0_size1.0",
    ]
    legends = [
        "DistilBERT A&Q",
        "DistilBERT S3&A&Q",
        "RoBERTa A&Q",
        "RoBERTa S3&A&Q",
    ]
    plot_training("train/loss", run_names, legends, type="c95", title="IT_train-loss")
    plot_training("eval/loss", run_names, legends, type="c95", title="IT_eval-loss")
    plot_training(
        "eval/accuracy", run_names, legends, type="c95", title="IT_eval-accuracy"
    )

    run_names = [
        "distilbert-base-uncased_S3&A&Q_0_r4_drop0.1_+0_size1.0",
        "roberta-base_S3&A&Q_0_r4_drop0.1_+0_size1.0",
        "google/gemma-2b_S3&A&Q_0_r4_drop0.1_+0_size1.0",
    ]
    legends = ["DistilBERT S3&A&Q", "RoBERTa S3&A&Q", "GEMMA-2B S3&A&Q"]
    plot_training(
        "train/loss", run_names, legends, type="c95", title="model_train-loss"
    )
    plot_training("eval/loss", run_names, legends, type="c95", title="model_eval-loss")
    plot_training(
        "eval/accuracy", run_names, legends, type="c95", title="model_eval-accuracy"
    )

    run_names = [
        "roberta-base_S3&A&Q_0_r4_drop0.1_+0_size1.0",
        "roberta-base_S3&A&Q_0_r4_drop0.1_+0_size0.75",
        "roberta-base_S3&A&Q_0_r4_drop0.1_+0_size0.5",
        "roberta-base_S3&A&Q_0_r4_drop0.1_+0_size0.25",
    ]
    legends = [
        "100%",
        "75%",
        "50%",
        "25%",
    ]
    plot_training("train/loss", run_names, legends, type="c95", title="size_train-loss")
    plot_training("eval/loss", run_names, legends, type="c95", title="size_eval-loss")
    plot_training(
        "eval/accuracy", run_names, legends, type="c95", title="size_eval-accuracy"
    )
