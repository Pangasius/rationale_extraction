import pandas as pd

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from processing.citation_dataset import compute_iou_mean


def TF_IDF_similarity(user_input, sentences, n):
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), lowercase=True)

    # Transform sentences to TF-IDF vectors
    X = vectorizer.fit_transform(sentences["sentence"])

    # Transform user input to TF-IDF vector
    input_vect = vectorizer.transform([user_input])

    # Compute cosine similarity between user input and sentences
    sentences["similarity"] = cosine_similarity(input_vect, X).flatten()

    # Sort sentences by similarity in descending order
    most_sim = sentences.sort_values(by="similarity", ascending=False)

    list_interval = []
    list_citation = []
    most_sim_index = most_sim.index.tolist()
    for i in range(n):
        if i >= len(most_sim_index):
            break
        # Get the index of the most similar sentence
        index = most_sim_index[i]

        # Compute start and end indices of the most similar sentence in the original text
        accumulator = 0
        for j in range(index):
            accumulator += (
                len(sentences.iloc[j]["sentence"]) + 1
            )  # Add length of sentence plus 1 for space
        start_index = accumulator
        end_index = accumulator + len(sentences.iloc[index]["sentence"])

        list_interval.append([start_index, end_index])
        list_citation.append(sentences.iloc[i]["sentence"])

    # Return the n most similar sentence and its start and end indices
    return list_citation, list_interval


def TF_IDF_similarity_threshold(user_input, sentences, threshold):
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), lowercase=True)

    # Transform sentences to TF-IDF vectors
    X = vectorizer.fit_transform(sentences["sentence"])

    # Transform user input to TF-IDF vector
    input_vect = vectorizer.transform([user_input])

    # Compute cosine similarity between user input and sentences
    sentences["similarity"] = cosine_similarity(input_vect, X).flatten()

    # Select only sentences above the threshold
    above_thres = sentences[sentences["similarity"] >= threshold]
    list_interval = []
    list_citation = []
    for index, row in above_thres.iterrows():
        # Compute start and end indices of the most similar sentence in the original text
        accumulator = 0
        for j in range(index):
            accumulator += (
                len(sentences.iloc[j]["sentence"]) + 1
            )  # Add length of sentence plus 1 for space
        start_index = accumulator
        end_index = accumulator + len(sentences.iloc[index]["sentence"])

        list_interval.append([start_index, end_index])
        list_citation.append(sentences.iloc[index]["sentence"])

    if not list_interval:
        list_interval.append([0, 0])
        list_citation.append("No citation found")

    # Return the n most similar sentence and its start and end indices
    return list_citation, list_interval


def citation_TF_IDF(answer, context, n):
    # Tokenize sentences in answer and context
    c_sentences = sent_tokenize(context)

    # Convert context sentences to a DataFrame
    c_sentences_df = pd.DataFrame(c_sentences, columns=["sentence"])

    # Initialize lists to store citations and their indices
    citations = []
    citations_index = []

    citations, citations_index = TF_IDF_similarity(answer, c_sentences_df, n)

    # Return lists of citations and their indices
    return citations, citations_index


def citation_TF_IDF_threshold(answer, context, threshold):
    # Tokenize sentences in answer and context
    c_sentences = sent_tokenize(context)

    # Convert context sentences to a DataFrame
    c_sentences_df = pd.DataFrame(c_sentences, columns=["sentence"])

    # Initialize lists to store citations and their indices
    citations = []
    citations_index = []

    citations, citations_index = TF_IDF_similarity_threshold(
        answer, c_sentences_df, threshold
    )

    # Return lists of citations and their indices
    return citations, citations_index


def get_reward(answer, context, ref_index, n):
    # Get citations and their indices using TF-IDF similarity
    _, citations_index = citation_TF_IDF(answer, context, n)

    # Compute IoU metric between citations and their corresponding context
    iou = compute_iou_mean([citations_index], ref_index)

    return iou
