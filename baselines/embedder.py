import pandas as pd
import nltk

from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from processing.citation_dataset import compute_iou_mean

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

nltk.download("punkt")


class Embedder:
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1", batch_size=1):
        self.model_name = model_name
        if self.model_name == "Salesforce/SFR-Embedding-Mistral":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Salesforce/SFR-Embedding-Mistral"
            )
            q_config = BitsAndBytesConfig(
                load_in_4bit=True,  # enable 4-bit quantization
                bnb_4bit_quant_type="nf4",  # information theoretically optimal dtype for normally distributed weights
                bnb_4bit_use_double_quant=True,  # quantize quantized weights //insert xzibit meme
                bnb_4bit_compute_dtype=torch.bfloat16,  # optimized fp format for ML
            )
            self.model = AutoModel.from_pretrained(
                "Salesforce/SFR-Embedding-Mistral", quantization_config=q_config
            )
            self.model.eval()
            self.batch_size = batch_size
        else:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    @staticmethod
    def truncate(text, max_length):
        if len(text) > max_length:
            text = text[:max_length]
        return text

    def sfr_embed(self, input_texts):
        max_length = 512
        input_texts = [self.truncate(text, max_length) for text in input_texts]
        batch_dict = self.tokenizer(
            input_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    def embedding_similarity(self, user_input, sentences, X, n):
        # Transform user input to embedding vector
        input_vect = None
        if self.model_name == "Salesforce/SFR-Embedding-Mistral":
            task = "Given a web search query, retrieve relevant passages that answer the query"
            user_input = self.get_detailed_instruct(task, user_input)
            input_vect = self.sfr_embed([user_input])
        else:
            input_vect = self.model.encode([user_input])

        if input_vect is None:
            print("Error: User input embedding failed")
            return None, None

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

            # Compute start and end indices of the most similar
            # sentence in the original text
            accumulator = 0
            for j in range(index):
                # Add length of sentence plus 1 for space
                accumulator += len(sentences.iloc[j]["sentence"]) + 1
            start_index = accumulator
            end_index = accumulator + len(sentences.iloc[index]["sentence"])

            list_interval.append([start_index, end_index])
            list_citation.append(sentences.iloc[i]["sentence"])

        # Return the n most similar sentence and its start and end indices
        return list_citation, list_interval

    def embedding_similarity_threshold(self, user_input, sentences, X, threshold):
        # Transform user input to embedding vector
        input_vect = None
        if self.model_name == "Salesforce/SFR-Embedding-Mistral":
            task = "Given a web search query, retrieve relevant passages that answer the query"
            user_input = self.get_detailed_instruct(task, user_input)
            input_vect = self.sfr_embed([user_input])
        else:
            input_vect = self.model.encode([user_input])

        if input_vect is None:
            print("Error: User input embedding failed")
            return None, None

        # Compute cosine similarity between user input and sentences
        sentences["similarity"] = cosine_similarity(input_vect, X).flatten()

        # Select only sentences above the threshold
        above_thres = sentences[sentences["similarity"] >= threshold]
        list_interval = []
        list_citation = []
        for index, row in above_thres.iterrows():
            # Compute start and end indices of the most similar
            # sentence in the original text
            accumulator = 0
            for j in range(index):
                # Add length of sentence plus 1 for space
                accumulator += len(sentences.iloc[j]["sentence"]) + 1
            start_index = accumulator
            end_index = accumulator + len(sentences.iloc[index]["sentence"])

            list_interval.append([start_index, end_index])
            list_citation.append(sentences.iloc[index]["sentence"])

        if not list_interval:
            list_interval.append([0, 0])
            list_citation.append("No citation found")

        # Return the n most similar sentence and its start and end indices
        return list_citation, list_interval

    @staticmethod
    def add_prefix_nomic(s):
        s = "search_query: " + s
        return s

    def citation_embedding(self, answer, context, n, progress_bar=None):
        # Tokenize sentences in answer and context
        c_sentences = sent_tokenize(context)

        # Convert context sentences to a DataFrame
        c_sentences_df = pd.DataFrame(c_sentences, columns=["sentence"])
        if self.model_name == "nomic-ai/nomic-embed-text-v1":
            c_sentences_df["sentence"] = c_sentences_df.apply(
                lambda row: self.add_prefix_nomic(row["sentence"]), axis=1
            )
        # Transform sentences to embedding vectors
        X = []

        if self.model_name == "Salesforce/SFR-Embedding-Mistral":
            # Split the context into chunks of 2 sentences to avoid gpu memory overflow
            i = 0
            while i < len(c_sentences_df):
                end_index = i + self.batch_size
                if len(c_sentences_df) <= end_index:
                    end_index = len(c_sentences_df)
                if i == 0:
                    X = self.sfr_embed(c_sentences_df["sentence"].tolist()[0:end_index])
                else:
                    new_embeddings = []
                    sentences = c_sentences_df["sentence"].tolist()[i:end_index]
                    new_embeddings = self.sfr_embed(sentences)
                    X = X + new_embeddings

                i += self.batch_size
        else:
            X = self.model.encode(c_sentences_df["sentence"].tolist())

        if X is None:
            print("Error: Context sentences embedding failed")
            return None, None

        citations, citations_index = self.embedding_similarity(
            answer, c_sentences_df, X, n
        )

        if progress_bar is not None:
            progress_bar.update(1)

        return citations, citations_index

    def citation_embedding_threshold(self, answer, context, progress_bar, threshold):
        # Tokenize sentences in answer and context
        c_sentences = sent_tokenize(context)

        # Convert context sentences to a DataFrame
        c_sentences_df = pd.DataFrame(c_sentences, columns=["sentence"])
        if self.model_name == "nomic-ai/nomic-embed-text-v1":
            c_sentences_df["sentence"] = c_sentences_df.apply(
                lambda row: self.add_prefix_nomic(row["sentence"]), axis=1
            )

        # Transform sentences to embedding vectors
        X = None
        if self.model_name == "Salesforce/SFR-Embedding-Mistral":
            # Split the context into chunks of 2 sentences to avoid gpu memory overflow
            i = 0
            while i < len(c_sentences_df):
                end_index = i + self.batch_size
                if len(c_sentences_df) <= end_index:
                    end_index = len(c_sentences_df)
                if i == 0:
                    X = self.sfr_embed(c_sentences_df["sentence"].tolist()[0:end_index])
                else:
                    new_embeddings = []
                    sentences = c_sentences_df["sentence"].tolist()[i:end_index]
                    new_embeddings = self.sfr_embed(sentences)
                    X = X + new_embeddings

                i += self.batch_size
        else:
            X = self.model.encode(c_sentences_df["sentence"].tolist())

        if X is None:
            print("Error: Context sentences embedding failed")
            return None, None

        citations, citations_index = self.embedding_similarity_threshold(
            answer, c_sentences_df, X, threshold
        )

        # Return lists of citations and their indices
        progress_bar.update(1)
        return citations, citations_index

    def get_reward(self, answer, context, ref_index, n):
        # Get citations and their indices using embedding similarity
        citations, citations_index = self.citation_embedding(answer, context, n)

        iou = compute_iou_mean(citations_index, ref_index)

        return iou
