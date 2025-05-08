from typing import Dict, List, Optional, Tuple

import rich

import regex as re

from torch import Tensor
from tqdm import tqdm

from llm_classifier import llm_classifier
from baselines import n_grams
from processing.citation_dataset import CitationDataset

import wandb

import torch

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from transformers import BitsAndBytesConfig

from transformers.tokenization_utils import PreTrainedTokenizer as PTT
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast as PTTF

from attention.learning_tokens_utils import (
    get_peft_config,
    get_prepared_model,
    get_device_map,
)

from attention.analyse_attention_utils import compute_rewards


def compute_evaluate_rewards(
    prediction: List[str], reference: List[str], metrics
) -> Dict[str, List[Dict[str, float]]]:
    """Computes the different metrics based on string outputs

    Args:
        prediction (List[str]): The predicted strings
        reference (List[str]): The reference strings
        metrics (_type_): The metrics to use

    Returns:
        Dict[str, List[Dict[str, float]]]: The computed metrics
    """
    rewards = {
        "meteor": [],
        "bleu": [],
        "rouge": [],
        #  "bertscore": []
    }

    for pred, ref in zip(prediction, reference):
        # Avoid dividing by 0 for bleu length
        if re.search(r"[\w\d]+", pred) is None:
            pred = "Nothing"

        rewards["meteor"].append(
            metrics[0].compute(predictions=[pred], references=[ref])
        )
        rewards["bleu"].append(metrics[1].compute(predictions=[pred], references=[ref]))
        rewards["rouge"].append(
            metrics[2].compute(predictions=[pred], references=[ref])
        )
        #  rewards["bertscore"].append(metrics[3].compute(predictions=pred,
        #                                          references=ref,
        #                                          lang="en"))

    return rewards


def get_rl_model(
    model_name: str,
    access_token: Optional[str],
    from_checkpoint: str,
    save_model: bool = False,
    version: int = 1,
):
    """Loading specific to RL model to produce a model with value head (for PPO).

    Args:
        model_name (str): The model name
        access_token (Optional[str]): HuggingFace read token
        from_checkpoint (str): The checkpoint to load
        save_model (bool, optional): Ensures attention can be saved. Defaults to False.
        version (int, optional): Different LoRA configs. Defaults to 1.

    Returns:
        Tuple: The tokenizer and model
    """
    rich.print("Loading model")

    tokenizer, model = get_prepared_model(
        model_name=model_name,
        from_checkpoint=from_checkpoint,
        access_token=access_token,
        loading_checkpoint=True,
        train=False,
        no_special_tokens=True,
        no_quant=True,
        save_attention=True,
    )

    rich.print("Loaded!\nMerging model")

    model = model.merge_and_unload()

    if save_model:
        rich.print("Merged!\nSaving model")

        save_to = "attention/models/trl/llm/" + model_name + "/" + from_checkpoint
        model.save_pretrained(save_to)
        tokenizer.save_pretrained(save_to)

        rich.print("Saved to", save_to)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    device_map = get_device_map(
        model_name=model_name, max_memory={0: "3GiB", 1: "10GiB", "cpu": "30GiB"}
    )

    peft_config = get_peft_config(model_name=model_name, rl=True, version=version)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model,
        token=access_token,
        attn_implementation="eager",
        is_trainable=True,
        quantization_config=quantization_config,
        device_map=device_map,
        peft_config=peft_config,
    )

    print(model)

    print(model.pretrained_model.print_trainable_parameters())

    return tokenizer, model


def get_rl_trainer(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: PTT | PTTF,
    dataset,
    seed: int,
    model_name: str = "",
    batch_size=64,
    tracker_args: Optional[Dict] = None,
):
    """Custom training with PPOTrainer

    Args:
        model (AutoModelForCausalLMWithValueHead): The model
        tokenizer (PTT | PTTF): The tokenizer
        dataset (_type_): The dataset to train on
        seed (int): The seed
        model_name (str, optional): The name of the model (to save in config). Defaults to "".
        batch_size (int, optional): The batch size. Defaults to 64.
        tracker_args (Optional[Dict], optional): wandb args. Defaults to None.

    Returns:
        _type_: _description_
    """
    config = PPOConfig(
        exp_name="RL",
        model_name=model_name,
        tracker_kwargs=tracker_args,
        seed=seed,
        reward_model=None,
        query_dataset=None,
        remove_unused_columns=False,
        batch_size=batch_size,
        mini_batch_size=1,
        log_with="wandb",
        kl_penalty="kl",
        max_grad_norm=1,
    )

    def collator(data):
        data_dict = {}

        for key in data[0]:
            if "input_ids" in key:
                data_dict[key] = [torch.Tensor(d[key]).to(torch.int32) for d in data]
            elif "attention_mask" in key:
                data_dict[key] = [torch.Tensor(d[key]).to(torch.bool) for d in data]
            else:
                data_dict[key] = [d[key] for d in data]

        return data_dict

    trainer = PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    return trainer


def add_rewards_together(
    rewards: List[Tensor],
    other_rewards: Dict[str, List[Dict[str, float]]],
    reward_type: str,
    penalties: List[Tensor]
):
    """Different methods of adding rewards to produce a final reward

    Args:
        rewards (List[Tensor]): The rewards from the attention
        other_rewards (Dict[str, List[Dict[str, float]]]): The other rewards
        reward_type (str): How to combine the rewards
        penalties (List[Tensor]): The penalties for the rewards

    Raises:
        ValueError: If the reward type is not implemented

    Returns:
        List[Tensor]: The combined rewards
    """
    if reward_type == "attn":
        return rewards

    if reward_type == "meteor+attn":
        for i in range(len(rewards)):
            rewards[i] = (rewards[i] + other_rewards["meteor"][i]["meteor"]) / 2 - 0.7
        return rewards

    if reward_type == "meteor+attn-pen":
        for i in range(len(rewards)):
            rewards[i] = (rewards[i] + other_rewards["meteor"][i]["meteor"]) / 2 - penalties[i] - 0.5
        return rewards

    if reward_type == "meteor+attn-fancy":
        # trying this arXiv:2402.00742
        sig = torch.nn.functional.logsigmoid
        min_sig = sig(Tensor([0.0]).to(rewards[0].device)) * 2
        max_sig = sig(Tensor([1.0]).to(rewards[0].device)) * 2
        for i in range(len(rewards)):
            rew = sig(rewards[i]) + sig(
                Tensor([other_rewards["meteor"][i]["meteor"]]).to(rewards[0].device)
            )
            rewards[i] = (rew - min_sig) / (max_sig - min_sig)
        return rewards

    if reward_type == "meteor":
        return [
            Tensor([other_rewards["meteor"][i]["meteor"]]).to(rewards[0].device)
            for i in range(len(rewards))
        ]

    if reward_type == "NG":
        return rewards

    if reward_type == "meteor+NG-fancy":
        # trying this arXiv:2402.00742
        sig = torch.nn.functional.logsigmoid
        min_sig = sig(Tensor([0.0]).to(rewards[0].device)) * 2
        max_sig = sig(Tensor([1.0]).to(rewards[0].device)) * 2
        for i in range(len(rewards)):
            rew = sig(rewards[i]) + sig(
                Tensor([other_rewards["meteor"][i]["meteor"]]).to(rewards[0].device)
            )
            rewards[i] = (rew - min_sig) / (max_sig - min_sig)
        return rewards

    if reward_type == "classify":
        return rewards

    if reward_type == "meteor+classify-fancy":
        sig = torch.nn.functional.logsigmoid
        min_sig = sig(Tensor([0.0]).to(rewards[0].device)) * 2
        max_sig = sig(Tensor([1.0]).to(rewards[0].device)) * 2
        for i in range(len(rewards)):
            rew = sig(rewards[i]) + sig(
                Tensor([other_rewards["meteor"][i]["meteor"]]).to(rewards[0].device)
            )
            rewards[i] = (rew - min_sig) / (max_sig - min_sig)
        return rewards

    raise ValueError(f"Unimplemented reward type {reward_type}")


def rl_train(
    trainer: PPOTrainer,
    tokenizer: PTT | PTTF,
    truncate: Tuple[int, int],
    metrics,
    layer: int,
    head: int,
    reward_type: str,
    evaluate_dataloader,
    epoch: int,
    rank: int,
    threshold: float,
    c_dataset: CitationDataset,
):
    """Rl training loop

    Args:
        trainer (PPOTrainer): The trainer
        tokenizer (PTT | PTTF): The tokenizer
        truncate (Tuple[int, int]): The max size of the input and of the total.
        metrics (_type_): The metrics to use
        layer (int): The layer to compute the attention on
        head (int): The head to compute the attention on
        reward_type (str): The type of reward to use
        evaluate_dataloader (_type_): The dataloader to evaluate on
        epoch (int): The number of epochs
        rank (int): The rank of the attention
        threshold (float): The threshold of the attention
        c_dataset (_type_): The citation dataset for the citation id

    Raises:
        ValueError: If the reward type is not implemented
    """
    gen_len = truncate[1] - truncate[0]

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "eos_token_id": None,
        "output_attentions": True,
        "return_dict_in_generate": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": gen_len,
    }

    classifier: llm_classifier.CC_classifier = None
    if "classify" in reward_type:
        print("Loading of classifier")
        run_name = "roberta-base_S3&A&Q_0_r4_drop0.1_+0_size1.0"
        model_dir = "llm_classifier/train_model/model/"
        token_dir = "llm_classifier/train_model/tokenizer/"
        classifier = llm_classifier.CC_classifier(
            "roberta-base",
            "S3&A&Q",
            model_dir + run_name,
            token_dir + run_name,
        )
        print("Success the Classifier is loaded")

    for epoch in range(epoch):
        for dataloader in (trainer.dataloader, evaluate_dataloader):
            for iteration, batch in tqdm(enumerate(dataloader)):
                query_tensors = batch["input_ids"]
                id_tensors = batch["id"]
                label_tensors = batch["labels"]

                #  Get response
                response_tensors = []
                #  response_masks = []
                rewards = []
                penalties = []
                labels = []

                batch["response"] = []
                batch["query"] = []
                batch["label"] = []

                for query, id_, label in zip(query_tensors, id_tensors, label_tensors):
                    labels.append(c_dataset[id_]["answer"])

                    response = trainer.generate(
                        query, batch_size=1, **generation_kwargs
                    )

                    ids = response["sequences"].squeeze()[
                        -len(response["attentions"]) :
                    ]

                    response_tensors.append(ids)

                    first_eos = (ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[
                        0
                    ]
                    if len(first_eos) > 0:
                        first_eos = first_eos[0] + 1
                    else:
                        first_eos = len(ids)

                    text = tokenizer.decode(ids[:first_eos], skip_special_tokens=True)
                    batch["response"].append(text)
                    batch["query"].append(
                        tokenizer.decode(query, skip_special_tokens=True)
                    )
                    batch["label"].append(
                        tokenizer.decode(label, skip_special_tokens=True)
                    )

                    if "attn" in reward_type or reward_type == "meteor":
                        reward_attention = (
                            torch.Tensor(
                                compute_rewards(
                                    attentions=response["attentions"][:first_eos],
                                    output_tokens=response["sequences"].squeeze()[
                                        :-len(response["attentions"]) + first_eos
                                    ] if first_eos < len(response["attentions"]) else response["sequences"].squeeze(),
                                    tokenizer=tokenizer,
                                    layers=[layer],
                                    heads=[head],
                                    attn_type="total",
                                    citation_dataset=c_dataset,
                                    dataset_id=id_,
                                    ranks=[rank],
                                    thresholds=[threshold],
                                )["1", layer, head, rank, threshold]
                            )
                            .mean()
                            .to(query.device)
                        )
                        penalties.append(torch.tensor([0.5 if first_eos == len(ids) else 0]).to(query.device))
                        rewards.append(reward_attention)

                    elif "NG" in reward_type:
                        row = c_dataset[id_]
                        context = row["context"]
                        citation = row["citation_index"]
                        reward = n_grams.get_reward(text, context, [citation], 1)
                        rewards.append(torch.tensor(reward).to(query.device))

                    elif "classify" in reward_type:
                        row = c_dataset[id_]
                        context = row["context"]
                        citation = row["citation_index"]
                        question = row["question"]
                        reward = classifier.compute_IoU(
                            text, question, context, citation
                        )
                        rewards.append(torch.tensor(reward).to(query.device))

                    else:
                        raise ValueError(f"Unimplemented reward type {reward_type}")

                # Compute all other rewards that might be useful
                other_rewards = compute_evaluate_rewards(
                    batch["response"], labels, metrics
                )

                other_rewards["reward_attention"] = [r.item() for r in rewards]

                rewards = add_rewards_together(rewards, other_rewards, reward_type, penalties)

                # If we're done we can just sum and send it
                to_log = {}
                for key, value in other_rewards.items():
                    for item in value:
                        if isinstance(item, float) or isinstance(item, Tensor):
                            if to_log.get(key) is None:
                                to_log[key] = item / len(value)
                            to_log[key] += item / len(value)
                        elif isinstance(item, dict):
                            if to_log.get(key) is None:
                                to_log[key] = {}
                            for k, v in item.items():
                                if isinstance(v, float):
                                    if to_log[key].get(k) is None:
                                        to_log[key][k] = v / len(value)
                                    to_log[key][k] += v / len(value)
                        else:
                            raise ValueError(f"Unimplemented type {type(item)}")

                if dataloader == trainer.dataloader:
                    wandb.log(to_log, commit=False)
                    wandb.log({"epoch": epoch})
                    #  Run PPO step
                    stats = trainer.step(query_tensors, response_tensors, rewards)

                    # wandb stresses out when it sees inf
                    stats["ppo/policy/ratio"][
                        abs(stats["ppo/policy/ratio"] - 1) > trainer.config.cliprange
                    ] = 0

                    trainer.log_stats(
                        stats,
                        batch,
                        rewards,
                        columns_to_log=("query", "label", "response"),
                    )
                else:
                    wandb.log({"evaluation": to_log})
