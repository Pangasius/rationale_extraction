import numpy as np
import nltk
from tqdm import tqdm

import matplotlib.pyplot as plt
from tabulate import tabulate
import math

import baselines.n_grams as ng
from baselines.embedder import Embedder

from scipy import stats

from processing.citation_dataset import CitationDataset, compute_iou_mean


AVAILABLE_METHODS = [
    "ngramsTopK",
    "embeddingTopK",
    "embeddingThreshold",
    "ngramsThreshold",
]
AVAILABLE_MODELS = ["ngrams", "bert", "nomic", "sfr"]


def compute_metrics(dataset, printed=False, title=""):
    dataset["IoU"] = dataset.apply(
        lambda row: compute_iou_mean([row["prediction_index"]], [row["citation_index"]]),
        axis=1,
    )
    if printed:
        print_metrics(dataset, title)
    return dataset, np.mean(dataset[dataset["IoU"].notna()]["IoU"])


def print_metrics(dataset, title, type="sub_questions"):
    # Iterate over the DataFrame row-wise
    clean_data = dataset[dataset["IoU"].notna()]
    if type == "sub_questions":
        score = [0 for i in range(5)]
        dist = [0 for i in range(5)]
        for _, row in clean_data.iterrows():
            dist[row["num_sub_question"] - 1] += 1
            score[row["num_sub_question"] - 1] += row["IoU"]

        dist_array = ["dist", np.sum(dist)]
        score_array = ["score", np.sum(score) / np.sum(dist)]
        type_score = np.array(score) / np.array(dist)
        score_array.extend(type_score)
        dist_array.extend(dist)

        table = [["", "Avg", "1-SQ", "2-SQ", "3-SQ", "4-SQ"], dist_array, score_array]
        print("\n\n", title, " :")
        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

    elif type == "num_sentences":
        nbr_sentences = [3, 6, 10, math.inf]

        score = [0 for _ in range(len(nbr_sentences))]
        dist = [0 for _ in range(len(nbr_sentences))]
        var = [[] for _ in range(len(nbr_sentences))]

        for _, row in clean_data.iterrows():
            cnt = len(nltk.sent_tokenize(row["context"]))
            for i in range(len(nbr_sentences)):
                if cnt <= nbr_sentences[i]:
                    dist[i] += 1
                    score[i] += row["IoU"]
                    var[i].append(row["IoU"])
                    break

        dist_array = ["dist", np.sum(dist)]
        score_array = ["score"]
        type_score = np.array(score) / np.array(dist)
        dist_array.extend(dist)

        # compute the student-t confidence interval

        def confidence_interval(data, confidence=0.95):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a), stats.sem(a)
            h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
            return m, h

        # total score
        m, h = confidence_interval(clean_data["IoU"])
        score_array.append(f"{m:.2f} ± {h:.2f}")

        for i in range(len(nbr_sentences)):
            m, h = confidence_interval(var[i])
            score_array.append(f"{m:.2f} ± {h:.2f}")

        table = [
            ["", "Avg", "1-3", "4-6", "7-10", "11+"],
            dist_array,
            score_array,
        ]
        print("\n\n", title, " :")
        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


def plot_IoU(series, title="IoU_box_plot"):
    # plot
    keys = []
    data = []
    for key, serie in series.items():
        data.append(serie.to_numpy())
        keys.append(key)

    fig, ax = plt.subplots()
    _ = ax.boxplot(
        data,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1},
        boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
        whiskerprops={"color": "C0", "linewidth": 1.5},
        capprops={"color": "C0", "linewidth": 1.5},
    )
    ax.set_xlabel("k")
    ax.set_ylabel("IoU")
    ax.grid()
    plt.savefig("baselines/plots/{}.png".format(title))


def fine_tune_threshold(
    dataset, model="ngrams", threshold_space=np.linspace(0.1, 0.9, 80), batch_size=4
):
    """
    Fine-tunes the threshold for the model on the dataset.

    Parameters:
        dataset (pandas.DataFrame): The dataset to fine-tune the threshold on.
        model (str): The model to fine-tune the threshold for.
        Default is "ngrams".

    Returns:
        float: The fine-tuned threshold.

    """
    print("Fine-tuning threshold for model: ", model)
    splits = dataset.to_splits(shuffle=True, seed=42)

    train_set, _, test_set = (
        splits["train"].to_pandas(),
        splits["val"].to_pandas(),
        splits["test"].to_pandas(),
    )

    # N-Grams Model fine-tuning with threshold #
    best_score = -1
    best_threshold = -1

    if model == "bert":
        embedder = Embedder(
            model_name="sentence-transformers/bert-large-nli-mean-tokens"
        )
    elif model == "nomic":
        embedder = Embedder(model_name="nomic-ai/nomic-embed-text-v1")
    elif model == "sfr":
        embedder = Embedder(
            model_name="Salesforce/SFR-Embedding-Mistral", batch_size=batch_size
        )
    elif model == "ngrams":
        pass
    else:
        raise ValueError("Invalid model name")

    for threshold in threshold_space:
        prediction_and_index = None
        if model == "ngrams":
            prediction_and_index = train_set.apply(
                lambda row: ng.citation_TF_IDF_threshold(
                    row["answer"], row["context"], threshold
                ),
                axis=1,
            )

        elif model == "bert":
            progress_bar = tqdm(total=train_set.shape[0], position=0, leave=True)
            prediction_and_index = train_set.apply(
                lambda row: embedder.citation_embedding_threshold(
                    row["answer"], row["context"], progress_bar, threshold
                ),
                axis=1,
            )

        elif model == "nomic":
            progress_bar = tqdm(total=train_set.shape[0], position=0, leave=True)
            prediction_and_index = train_set.apply(
                lambda row: embedder.citation_embedding_threshold(
                    row["answer"], row["context"], progress_bar, threshold
                ),
                axis=1,
            )
        elif model == "sfr":
            progress_bar = tqdm(total=train_set.shape[0], position=0, leave=True)
            prediction_and_index = train_set.apply(
                lambda row: embedder.citation_embedding_threshold(
                    row["answer"], row["context"], progress_bar, threshold
                ),
                axis=1,
            )

        # Extract citations and their indices from prediction_and_index
        if prediction_and_index is not None:
            train_set["prediction"] = prediction_and_index.apply(lambda x: x[0])
            train_set["prediction_index"] = prediction_and_index.apply(lambda x: x[1])

        # Compute evaluation metrics
        train_set, score = compute_metrics(train_set)
        print("Threshold: ", threshold, "\tScore: ", score)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    evaluate_threshold(test_set, best_threshold, model)

    return best_threshold


def evaluate_threshold(dataset, threshold, model="ngrams", batch_size=4):
    """
    Evaluates the performance of the model on the dataset.

    Parameters:
        dataset (pandas.DataFrame): The dataset to evaluate.
        threshold (float): The threshold to use for the evaluation.

    Returns:
        pandas.DataFrame: The evaluation results.

    """
    print(model, threshold)
    if model == "ngrams":
        # N-Grams Model evaluation with threshold #
        prediction_and_index = dataset.apply(
            lambda row: ng.citation_TF_IDF_threshold(
                row["answer"], row["context"], threshold
            ),
            axis=1,
        )
    elif model == "bert":
        progress_bar = tqdm(total=dataset.shape[0], position=0, leave=True)
        embedder = Embedder(
            model_name="sentence-transformers/bert-large-nli-mean-tokens"
        )
        prediction_and_index = dataset.apply(
            lambda row: embedder.citation_embedding_threshold(
                row["answer"], row["context"], progress_bar, threshold
            ),
            axis=1,
        )
    elif model == "nomic":
        progress_bar = tqdm(total=dataset.shape[0], position=0, leave=True)
        embedder = Embedder(model_name="nomic-ai/nomic-embed-text-v1")
        prediction_and_index = dataset.apply(
            lambda row: embedder.citation_embedding_threshold(
                row["answer"], row["context"], progress_bar, threshold
            ),
            axis=1,
        )

    elif model == "sfr":
        progress_bar = tqdm(total=dataset.shape[0], position=0, leave=True)
        embedder = Embedder(
            model_name="Salesforce/SFR-Embedding-Mistral", batch_size=batch_size
        )
        prediction_and_index = dataset.apply(
            lambda row: embedder.citation_embedding_threshold(
                row["answer"], row["context"], progress_bar, threshold
            ),
            axis=1,
        )

    else:
        print("Invalid model name the model supported are ngrams, bert, nomic")
        raise ValueError("Invalid model name")

    # Extract citations and their indices from prediction_and_index
    dataset["prediction"] = prediction_and_index.apply(lambda x: x[0])
    dataset["prediction_index"] = prediction_and_index.apply(lambda x: x[1])

    # Compute evaluation metrics
    dataset, score = compute_metrics(dataset)
    if model == "ngrams":
        print_metrics(
            dataset,
            "Benchmarking results for N-grams model with threshold " + str(threshold),
            "num_sentences",
        )
    elif model == "bert":
        print_metrics(
            dataset,
            "Benchmarking results for Bert Embedding model with threshold "
            + str(threshold),
            "num_sentences",
        )
    elif model == "nomic":
        print_metrics(
            dataset,
            "Benchmarking results for Nomic Embedding model with threshold "
            + str(threshold),
            "num_sentences",
        )
    elif model == "sfr":
        print_metrics(
            dataset,
            "Benchmarking results for SFR Embedding model with threshold "
            + str(threshold),
            "num_sentences",
        )
    return dataset


def evaluate_embedding(dataset, model_name, k=1, title="", batch_size=4):
    """
    Evaluates the performance of the model on the dataset.

    Parameters:
        dataset (pandas.DataFrame): The dataset to evaluate.
        model_name (str): The name of the embedding model to use.
        k (int): The number of top sentences to consider.
        title (str): The title for the evaluation results.

    Returns:
        pandas.DataFrame: The evaluation results.

    """
    embedder = None
    if model_name == "sfr":
        print("loading sfr model")
        print("batch size: ", batch_size)
        embedder = Embedder(
            model_name="Salesforce/SFR-Embedding-Mistral", batch_size=batch_size
        )
    if model_name == "bert":
        embedder = Embedder(
            model_name="sentence-transformers/bert-large-nli-mean-tokens"
        )
    if model_name == "nomic":
        embedder = Embedder(model_name="nomic-ai/nomic-embed-text-v1")

    if embedder is None:
        print("failed to load the model")
        return

    progress_bar = tqdm(total=dataset.shape[0], position=0, leave=True)

    prediction_and_index = dataset.apply(
        lambda row: embedder.citation_embedding(
            row["answer"], row["context"], k, progress_bar
        ),
        axis=1,
    )

    # Extract citations and their indices from prediction_and_index
    dataset["prediction"] = prediction_and_index.apply(lambda x: x[0])
    dataset["prediction_index"] = prediction_and_index.apply(lambda x: x[1])

    print(dataset["prediction"].head())
    print(dataset["prediction_index"].head())

    # Compute evaluation metrics
    dataset, score = compute_metrics(dataset)
    print_metrics(dataset, title, "num_sentences")


def evaluate_TopK_embedding(
    dataset, model_name, k=[1, 2, 3, 4, 5], plot=False, batch_size=4
):
    """
    Evaluates the performance of the model on the dataset.

    Parameters:
        dataset (pandas.DataFrame): The dataset to evaluate.
        model_name (str): The name of the embedding model to use.
        k (int): The number of top sentences to consider.
        plot (bool): Whether to plot the evaluation results.

    Returns:
        pandas.DataFrame: The evaluation results.

    """
    IoU_series = {}
    for i in k:
        title = "Benchmarking results for {} top-{} sentences".format(model_name, i)
        evaluate_embedding(dataset, model_name, i, title, batch_size)
        IoU_series[i] = dataset["IoU"]

    if plot:
        title = "IoU_box_plot_{}".format(model_name)
        plot_IoU(IoU_series, title)


def evaluate_NGrams(dataset, k, title=""):
    """
    Evaluates the performance of the model on the dataset.

    Parameters:
        dataset (pandas.DataFrame): The dataset to evaluate.
        k (int): The number of top sentences to consider.
        title (str): The title for the evaluation results.

    Returns:
        pandas.DataFrame: The evaluation results.

    """
    prediction_and_index = dataset.apply(
        lambda row: ng.citation_TF_IDF(row["answer"], row["context"], k), axis=1
    )

    # Extract citations and their indices from prediction_and_index
    dataset["prediction"] = prediction_and_index.apply(lambda x: x[0])
    dataset["prediction_index"] = prediction_and_index.apply(lambda x: x[1])

    # Compute evaluation metrics
    dataset, score = compute_metrics(dataset)
    print_metrics(dataset, title, "num_sentences")
    return dataset, score


def evaluate_TopK_NGrams(dataset, k=[1, 2, 3, 4, 5], plot=False):
    """
    Evaluates the performance of the model on the dataset.

    Parameters:
        dataset (pandas.DataFrame): The dataset to evaluate.
        k (int): The number of top sentences to consider.
        plot (bool): Whether to plot the evaluation results.

    Returns:
        pandas.DataFrame: The evaluation results.
    """
    IoU_series = {}
    for i in k:
        title = "Benchmarking results for N-grams top-{} sentences".format(i)
        proces_dataset, _ = evaluate_NGrams(dataset, i, title)
        IoU_series[i] = proces_dataset["IoU"]

    if plot:
        plot_IoU(IoU_series, "IoU_box_plot_NGrams")


def dataset_distribution(dataset, type="sub_questions", title="Dataset distribution"):
    if type == "sub_questions":
        # Iterate over the DataFrame row-wise
        dist = [0 for i in range(5)]
        for index, row in dataset.iterrows():
            dist[row["num_sub_question"] - 1] += 1
        cmap = plt.cm.tab10
        colors = cmap(range(5))

        fig, ax = plt.subplots()
        ax.bar(range(1, 6), dist, width=1, color=colors, linewidth=0.7)
        ax.set(xlim=(0, 6), xticks=np.arange(1, 6))
        ax.set_xlabel("Number of sub-questions")
        ax.set_ylabel("Number of data-points")
        ax.grid()
        plt.savefig("baselines/plots/{}.png".format(title))

    elif type == "context_length":
        dataset["Context_length"] = dataset.apply(
            lambda row: len(row["context"].split()), axis=1
        )
        plt.figure()
        dataset["Context_length"].hist(bins=50)
        plt.xlabel("Length of context")
        plt.ylabel("Number of data-points")
        plt.grid()
        plt.savefig("baselines/plots/{}.png".format(title))

    elif type == "number_sentences":
        dataset["Number_sentences"] = dataset.apply(
            lambda row: len(nltk.sent_tokenize(row["context"])), axis=1
        )
        plt.figure()
        dataset["Number_sentences"].hist(bins=20)
        plt.xlabel("Number of sentences", fontsize=12)
        plt.ylabel("Number of examples", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid()
        plt.savefig("baselines/plots/{}.png".format(title))


def ensemble_distribution(
    train_dataset, test_dataset, eval_dataset, title="Ensemble distribution"
):

    num_sentences = [3, 6, 10, math.inf]

    dists = np.zeros((3, len(num_sentences)))

    for dataset, dist in zip([train_dataset, test_dataset, eval_dataset], dists):
        for _, row in dataset.iterrows():
            cnt = len(nltk.sent_tokenize(row["context"]))
            for i in range(len(num_sentences)):
                if cnt <= num_sentences[i]:
                    dist[i] += 100 / len(dataset)
                    break

    species = ("1-3", "4-6", "7-10", "11+")
    dist_means = {
        r"Train": dists[0],
        r"Eval": dists[1],
        r"Test": dists[2],
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained", figsize=(7, 7))

    for attribute, measurement in dist_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r"Ratio of dataset $[\%]$", fontsize=15)
    ax.set_xticks(x + width, species, fontsize=15)
    ax.set_xlabel(fontsize=15, xlabel="Number of sentences")
    ax.legend(loc="upper left", ncols=3, fontsize=13)
    ax.grid()

    plt.yticks(fontsize=15)
    plt.savefig("baselines/plots/dataset_dist_ensemble.pdf")


def evaluate():
    dataset = CitationDataset("processing/annotated_datasets/clear_dataset.jsonl")

    # dataset_distribution(dataset, "number_sentences", "Number of sentences distribution")

    # evaluate_TopK_NGrams(test_set, plot=True)

    # evaluate_TopK_embedding(test_set, "sentence-transformers/bert-large-nli-mean-tokens", plot=True)

    # evaluate_TopK_embedding(test_set, 'Salesforce/SFR-Embedding-Mistral', plot=True)

    fine_tune_threshold(dataset, "sfr", np.linspace(0.71, 0.74, 10), 8)
