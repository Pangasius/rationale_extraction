from typing import Dict, List, Tuple

import regex as re

import math

import pandas as pd

from tqdm import tqdm

import nltk

import torch
from torch import Tensor

from torch.utils.data import DataLoader

from transformers.tokenization_utils import PreTrainedTokenizer as PTT
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast as PTTF

from processing.citation_dataset import CitationDataset, compute_iou_list

QUESTION_SIG = "### Question:"
CONTEXT_SIG = "### Context:"
ANSWER_SIG = "### Answer:"
END_SIG = "### End"


class ATTN_TYPE:
    TOTAL = "total"
    CITATION = "citation_index"
    POST= "post"


def get_citation_id(sequence: List[int], tokenizer: PTT | PTTF) -> tuple[str, int]:
    """
    Decode the number in between the citation open and close tokens.

    Args:
    - sequence (List[int]): starting from the citation open token + 1
    - tokenizer (PTT | PTTF): encodes the citation tokens

    Raises:
    - ValueError: if the close token is more than one token,
    meaning the tokenization is incorrect

    Returns:
    - citation_id: str
    - length: int
    """

    close = tokenizer.encode("<citation_close>", add_special_tokens=False)

    if len(close) > 1:
        raise ValueError("Close token is more than one token")

    citation_id = ""
    length = 1

    for j in range(0, len(sequence)):
        if sequence[j] == close[0]:
            break
        citation_id += tokenizer.decode(sequence[j])
        length += 1

    return citation_id, length


def find_citation_spans(
    sequence: List[int], tokenizer: PTT | PTTF
) -> dict[str, tuple[int, int]]:
    """Creates a dictionary of citation spans in the sequence.
    Each span is identified by the citation id and contains
    the bounds of the citation in the sequence.

    Args:
        sequence (List[int]): List of token ids
        tokenizer (PTT | PTTF): PTT | PTTF object

    Raises:
        ValueError: If the start or end token is more than one token,
        meaning the tokenization is incorrect

    Returns:
        dict: Dictionary of citation spans
    """

    start = tokenizer.encode("<start_citation_open>", add_special_tokens=False)
    end = tokenizer.encode("<end_citation_open>", add_special_tokens=False)

    if len(start) > 1 or len(end) > 1:
        raise ValueError("Start or end token is more than one token")

    spans = {}
    for i, token in enumerate(sequence):
        if token == start[0]:
            # Identify the citation id
            c_id, length = get_citation_id(sequence[i + 1 :], tokenizer)

            if c_id in spans:
                # print("Duplicate citation id", c_id)
                break

            for j in range(i + length, len(sequence)):
                if sequence[j] == end[0]:
                    c_id_end, length_end = get_citation_id(sequence[j + 1 :], tokenizer)

                    if c_id == c_id_end:
                        spans[c_id] = (i + 1 + length, j - 1)
                        break

                    j += length_end

            if c_id not in spans:
                # print("Unmatched citation id", c_id)
                continue

    return spans


def attn_mean(
    attn: List[List[Tensor]],
    layers: List[int],
    heads: List[int],
    answer_begin: int,
) -> Dict[Tuple[int, int], Tensor]:
    """Given a returned attention tensor, extract the attention
    at a specific token index for all layers and heads.

    Args:
        attn (List[List[Tensor]]): List of attention tensors
        layers (List[int]): List of layer indices
        heads (List[int]): List of head indices
        answer_begin (int): The index of the answer beginning

    Returns:
        Tensor: Attention values at the token index
    """

    num_tokens = len(attn)

    return_dict = {}
    for layer in layers:
        for head in heads:
            for token in range(num_tokens):
                if return_dict.get((layer, head)) is None:
                    return_dict[(layer, head)] = torch.zeros(
                        answer_begin,
                        dtype=torch.float16,
                        device=attn[token][layer].device,
                    )

                return_dict[(layer, head)] += (
                    attn[token][layer][0, head, -1, :answer_begin] / num_tokens
                )

    return return_dict

def attn_mean_post(
    attn: List[List[Tensor]],
    layers: List[int],
    heads: List[int],
    answer_begin: int,
) -> Dict[Tuple[int, int], Tensor]:
    """Given a returned attention tensor, extract the attention
    at a specific token index for all layers and heads.

    Args:
        attn (List[List[Tensor]]): List of attention tensors
        layers (List[int]): List of layer indices
        heads (List[int]): List of head indices
        answer_begin (int): The index of the answer beginning

    Returns:
        Tensor: Attention values at the token index
    """

    return_dict = {}
    for layer in layers:
        for head in heads:
            return_dict[(layer, head)] = (
                attn[layer][0, head, answer_begin:, :answer_begin].mean(dim=0)
            )

    return return_dict


def attn_for_spans(
    attn: List[List[Tensor]],
    layers: List[int],
    heads: List[int],
    sequence: List[int],
    tokenizer: PTT | PTTF,
) -> Dict[str, Dict[Tuple[int, int], Tensor]]:
    """Computes the average attention for each citation span in the sequence.

    Args:
        attn (List[List[Tensor]]): the attention tensors
        layers (List[int]): List of layer indices
        heads (List[int]): List of head indices
        sequence (List[int]): the token ids
        tokenizer (PTT | PTTF): the tokenizer object

    Returns:
        dict: Dictionary of citation id to attention values
    """
    spans = find_citation_spans(sequence, tokenizer)

    answer_begin = attn[0][0].shape[2]

    att_dict = {}
    for c_id, (start, end) in spans.items():
        start = start - answer_begin
        end = end - answer_begin

        att_dict[c_id] = attn_mean(attn[start:end], layers, heads, answer_begin)

    return att_dict


def attn_per_character(
    tokens_str: List[str], tokenizer: PTT | PTTF
) -> tuple[Dict[int, Tuple[int, int]], str]:
    """Assigns a weight to each character in the sequence.

    Args:
        tokens_str (List[str]): The strings of the tokens
        tokenizer (PTT | PTTF): The tokenizer

    Returns:
        tuple[List[float], str]: The weights for each character and the associated sentence
    """

    sentence = ""
    sentence_before = ""
    weight_map = {}

    start = -1
    for index, token in enumerate(tokens_str):
        # Skip padding and eos tokens
        if start == -1 and (
            token == tokenizer.pad_token or token == tokenizer.bos_token
        ):
            continue

        token = token.replace("▁", " ")

        if start == -1:
            sentence_before += token

            if CONTEXT_SIG not in sentence_before[-len(CONTEXT_SIG) * 2 :]:
                continue

            start = index
            continue

        # register to weight map
        weight_map.update({len(sentence) + i: (index, 1) for i in range(len(token))})

        sentence += token

        if ANSWER_SIG in sentence[-len(ANSWER_SIG) * 2 :]:
            sig = sentence.rfind(ANSWER_SIG)
            sentence = sentence[:sig]
            break

    return weight_map, sentence


def get_weight_dict(
    attn: List[List[Tensor]],
    sequence: List[int],
    tokenizer: PTT | PTTF,
    layers: List[int],
    heads: List[int],
    attn_type: str,
) -> Dict[str, Dict[Tuple[int, int], Tensor]]:
    """Get the weights for the attention values as a dictionary.

    Args:
        attn (List[List[Tensor]]): the attention tensors
        sequence (List[int]): the token ids
        tokenizer (PTT | PTTF): the tokenizer
        layers (List[int]): List of layer indices
        heads (int): List of head indices
        attn_type (str): the type of attention to compute

    Raises:
        ValueError: If the attention type is not supported

    Returns:
        dict: Dictionary of weights for each token
    """

    if attn_type == ATTN_TYPE.TOTAL:
        weights = {"1": attn_mean(attn, layers, heads, len(sequence) - len(attn))}
    elif attn_type == ATTN_TYPE.CITATION:
        weights = attn_for_spans(attn, layers, heads, sequence, tokenizer)
    elif attn_type == ATTN_TYPE.POST:
        weights = {"1": attn_mean_post(attn, layers, heads, attn[0].shape[2] - len(sequence))}
    else:
        raise ValueError("attn_type must be either 'total' or 'citation'")

    return weights


def compute_rewards(
    attentions: List[List[Tensor]],
    input_tokens: List[int],
    output_tokens: List[int],
    tokenizer: PTT | PTTF,
    layers: List[int],
    heads: List[int],
    attn_type: str,
    thresholds: List[float],
    ranks: List[int],
    citation_dataset: CitationDataset,
    dataset_id: int,
) -> Dict[Tuple[str, int, int, int, float], List[float]]:
    """Finds the sentences with the highest attention weights.
    Can filter out sentences with weights below a threshold.
    Can keep only the top rank sentences.

    Args:
        attentions (List[List[Tensor]]): the attention tensors
        output_tokens (List[int]): the token ids
        tokenizer (PTT | PTTF): the tokenizer
        layers (List[int]): List of layer indices
        heads (List[int]): List of head indices
        attn_type (str): the type of attention to compute
        thresholds (List[float]): The minimum weight to keep a sentence.
        ranks (List[int]): The number of sentences to keep (top-k).
        dataset_id (int): The dataset id of the data

    Returns:
        dict: Dictionary of sentences with their weights
    """

    strings = tokenizer.convert_ids_to_tokens(input_tokens)

    if isinstance(strings, str):
        strings = [strings]
        
    character_map, sentence_total = attn_per_character(strings, tokenizer)

    weights_dict = get_weight_dict(
        attentions, output_tokens, tokenizer, layers, heads, attn_type
    )

    sentences = nltk.sent_tokenize(sentence_total)
    sentences_dict = {}

    for weights_key, weights in weights_dict.items():
        for layer in layers:
            for head in heads:
                weight = weights[(layer, head)]

                sentences_weight = [None] * len(sentences)
                for sentence_index, sentence in enumerate(sentences):
                    index_start = sentence_total.find(sentence)
                    index_end = index_start + len(sentence) - 1

                    weight_start = character_map[index_start][0]
                    weight_end = character_map[index_end][0]

                    sentences_weight[sentence_index] = (
                        weight[weight_start:weight_end].mean(),
                        index_start,
                        index_end,
                    )

                for rank in ranks:
                    if rank < math.inf and rank > 0:
                        new_sentences_weight = sorted(
                            sentences_weight, reverse=True, key=lambda x: x[0]
                        )[:rank]
                    else:
                        new_sentences_weight = sentences_weight.copy()

                    for threshold in thresholds:
                        new_sentences_weight_2 = [
                            sentence_weight
                            for sentence_weight in new_sentences_weight
                            if sentence_weight[0] > threshold
                        ]

                        bounds = [
                            [
                                list(sentence_weight[1:3])
                                for sentence_weight in new_sentences_weight_2
                            ]
                        ]

                        iou = compute_iou_list(
                            bounds, citation_dataset[dataset_id]["sub_citation_index"]
                        )

                        sentences_dict[(weights_key, layer, head, rank, threshold)] = (
                            iou
                        )
    return sentences_dict


def colorize_tokens(
    weights: dict[str, dict[Tuple[int, int], Tensor]],
    tokens: List[str],
    tokenizer: PTT | PTTF,
    special_tokens: List[str],
    maxi: int,
    layer: int = 0,
    head: int = 0,
):
    """
    Colorizes the tokens based on the attention weights.
    Not used since a lot of changes have been made, be careful if using.

    Args:
        weights (dict[str, dict[Tuple[int, int], Tensor]]): the attention weights
        tokens (List[str]): the tokens whose attention weights are to be colorized
        tokenizer (PTT | PTTF): the tokenizer
        special_tokens (List[str]): the special tokens to be specially colored
        maxi (int): maximum number of tokens
        layer (int, optional): the layer to find in the weights. Defaults to 0.
        head (int, optional): the head to find in the weights. Defaults to 0.
    """
    def color_for(value: int, profile: int):
        # Maps value to RGB color depending on profile
        profiles = {
            1: (0, 0),
            2: (1, 0),
            3: (2, 0),
            4: (1, 1),
            5: (1, 2),
            6: (2, 2),
        }
        if profile in profiles:
            channel1, channel2 = profiles[profile]
            channels = ["00"] * 3
            channels[channel1] = f"{value:02X}"
            channels[channel2] = f"{value:02X}"
            R, G, B = channels
            return "".join([R, G, B])
        else:
            raise ValueError("Profile must be between 1 and 6. Profile: ", profile)

    colorized_tokens = []
    for i, token in enumerate(tokens):
        if token == tokenizer.pad_token or token == tokenizer.bos_token:
            continue

        if token == tokenizer.eos_token:
            if i > maxi:
                break
            continue

        token = token.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n")

        if token == "":
            continue

        if token in special_tokens:
            colorized_tokens.append(f"[bold][#C0C0FF]{token}[/][/bold]")
            continue

        profile = 1
        max_value = -math.inf
        for profile_key in range(1, 7):
            if weights.get(str(profile)) is None:
                continue
            if weights[str(profile)][(layer, head)][i] > max_value:
                max_value = weights[str(profile)][(layer, head)][i]
                profile = profile_key

        weight_color = int(
            max(min(64 + 127 * weights[str(profile)][(layer, head)][i].item(), 255), 0)
        )
        colorized_tokens.append(
            f"[bold][#CCCCCC on #{color_for(weight_color, profile + 1)}]{token}[/][/bold]"
        )

    regex_special = re.compile(r"<[/]?(.*?)>")
    colorized_tokens = [
        re.sub(regex_special, r"[bold][#C0C0FF]\1[/][/bold]", token)
        for token in colorized_tokens
    ]

    return "".join(colorized_tokens)


def get_colorized_text(
    attn: List[List[Tensor]],
    sequence: List[int],
    layers: List[int],
    heads: List[int],
    tokenizer: PTT | PTTF,
    special_tokens: List[str],
    attn_type="total",
) -> str:
    """
    Get the colorized text corresponding to the concatenation
    of the tokens in the sequence with the attention weights.
    Not used since a lot of changes have been made, be careful if using.
    """

    strings = tokenizer.convert_ids_to_tokens(sequence)

    if isinstance(strings, str):
        strings = [strings]

    weights = get_weight_dict(attn, sequence, tokenizer, layers, heads, attn_type)

    for layer in layers:
        for head in heads:
            text = ""
            text += f"Layer {layer}, Head {head}\n"
            text += colorize_tokens(
                weights, strings, tokenizer, special_tokens, len(sequence), layer, head
            )
            text += "\n\n"

    return text


def analyse_attention(
    tokenizer: PTT | PTTF,
    model: torch.nn.Module,
    path: str,
    name: str,
    layers: List[int],
    heads: List[int],
    ranks: List[int],
    thresholds: List[float],
    truncate: Tuple[int, int],
    dataloader: DataLoader,
    c_dataset: CitationDataset,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the model on the dataloader and computes the attention weights for each layer, head, rank and threshold.

    Args:
        tokenizer (PTT | PTTF): the tokenizer
        model (torch.nn.Module): the model
        path (str): path to save the results
        name (str): name of the file to save the results
        layers (List[int]): List of layer indices
        heads (List[int]): List of head indices
        ranks (List[int]): List of ranks (top-k)
        thresholds (List[float]): List of thresholds
        truncate (Tuple[int, int]): Maximum lengths of the input and total tokens
        attn_type (str): the type of attention to compute (total or citation)
        dataloader (DataLoader): the dataloader
        c_dataset (CitationDataset): the citation dataset

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The iou and answer dataframes (saved to path)
    """
    iou_dataframe = None

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            
            total_input = torch.cat((batch["input_ids"][0], batch["labels"][0]), dim = -1)

            response = model(
                input_ids=total_input.unsqueeze(0).to(model.device),
                output_attentions=True
            )

            dict_iou = compute_rewards(
                attentions=response["attentions"],
                input_tokens=batch["input_ids"][0].to(model.device),
                output_tokens=batch["labels"][0].to(model.device),
                tokenizer=tokenizer,
                layers=layers,
                heads=heads,
                attn_type=ATTN_TYPE.POST,
                citation_dataset=c_dataset,
                dataset_id=batch["id"][0],
                ranks=ranks,
                thresholds=thresholds,
            )

            for key, value in dict_iou.items():
                iou_entry = pd.DataFrame(
                    {
                        "layer": key[1],
                        "head": key[2],
                        "rank": key[3],
                        "threshold": key[4],
                        "iou": value,
                    }
                )
                iou_entry["dataset_id"] = batch["id"][0]

                iou_dataframe = pd.concat([iou_dataframe, iou_entry], ignore_index=True)

    iou_dataframe.to_csv(path + "/iou-" + name, index=False)

    return iou_dataframe
