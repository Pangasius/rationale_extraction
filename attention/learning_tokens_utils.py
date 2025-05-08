from typing import (
    List,
    Tuple,
    Optional,
    Dict,
)

import torch

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from accelerate import init_empty_weights, infer_auto_device_map

from transformers.tokenization_utils import PreTrainedTokenizer as PTT
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast as PTTF

from datasets import Dataset

# from transformers.trainer_pt_utils import AcceleratorConfig

from peft import (
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)

from pandas import DataFrame
import pickle as pkl

from tqdm import tqdm

import regex as re

from processing.citation_dataset import SPECIAL_TOKENS

QUESTION_SIG = "### Question:"
CONTEXT_SIG = "### Context:"
ANSWER_SIG = "### Answer:"
END_SIG = "### End"


def formatting_func(
    examples: Dict[str, List[str] | str], tokenizer: PTT | PTTF
) -> Tuple[List[str], List[str]]:
    """Formats the dataset entries to be used by the model

    Args:
        examples (Dict[str, List[str] | str]): A dictionary containing the
        keys "question", "context" and "answer" with the corresponding values
        tokenizer (PTT | PTTF): The tokenizer to be used

    Returns:
        Tuple[List[str], List[str]]: A Tuple containing the formatted inputs
    """
    if isinstance(examples["question"], str):
        # Ifa single object is passed, convert it to a List
        examples = {k: [v] for k, v in examples.items()}

    inputs = [
        QUESTION_SIG + question + "\n\n" + CONTEXT_SIG + context + "\n\n" + ANSWER_SIG
        for question, context in zip(examples["question"], examples["context"])
    ]

    targets = [answer + END_SIG + tokenizer.eos_token for answer in examples["answer"]]

    return inputs, targets


def preprocess_function(
    examples: Dict[str, List[str] | str],
    tokenizer: PTT | PTTF,
    truncate: Tuple[int, int],
    out_labels: bool = True,
    padding=True,
):
    """
    Makes the examples fit for Trainer.
    If out_labels is True, the function returns a dictionary with the keys
    "input_ids", "attention_mask" and "labels" where the labels are the targets
    If out_labels is False, the function returns a dictionary with the keys
    "input_ids", "attention_mask" and "labels" where the labels are the same

    Args:
        examples (Dict[str, List[str]  |  str]): A dictionary containing the
        keys "question", "context" and "answer" with the corresponding values
        tokenizer (PTT | PTTF): The tokenizer to be used
        truncate (Tuple[int, int]): The length of the input and total length
        out_labels (bool, optional): Whether to make targets different from
        input.
        Defaults to True.

    Returns:
        Dict: A dictionary containing the formatted inputs
    """

    inputs, targets = formatting_func(examples, tokenizer)

    if out_labels:
        if padding:
            # Make sure we pad on the correct side
            tokenizer.padding_side = "left"
            input_tokens = tokenizer(
                inputs, padding="max_length", truncation=True, max_length=truncate[0]
            )

            # Make sure we pad on the correct side
            tokenizer.padding_side = "right"
            target_tokens = tokenizer(
                targets,
                padding="max_length",
                truncation=True,
                max_length=1 + truncate[1] - truncate[0],
            )

            # remove bos token
            target_tokens = {k: [vv[1:] for vv in v] for k, v in target_tokens.items()}

            # Reset padding side
            tokenizer.padding_side = "left"
        else:
            input_tokens = tokenizer(inputs)

            target_tokens = tokenizer(targets)

        return {**input_tokens, "labels": target_tokens["input_ids"]}

    else:
        joined = [x + y for x, y in zip(inputs, targets)]

        input_tokens = tokenizer(joined)

        block_size = truncate[1]

        def group_texts(examples):
            concatenated_examples = {
                "input_ids": [],
                "attention_mask": [],
            }
            new_examples = {
                "input_ids": [],
                "attention_mask": [],
            }
            for exs, mask in zip(examples["input_ids"], examples["attention_mask"]):
                if len(exs) > block_size:
                    # skip too long examples
                    continue

                elif len(new_examples["input_ids"]) + len(exs) > block_size:
                    padding_length = block_size - len(new_examples["input_ids"])

                    # insert on the left side
                    new_examples["input_ids"] = [
                        tokenizer.pad_token_id
                    ] * padding_length + new_examples["input_ids"]
                    new_examples["attention_mask"] = [
                        0
                    ] * padding_length + new_examples["attention_mask"]

                    concatenated_examples["input_ids"].append(new_examples["input_ids"])
                    concatenated_examples["attention_mask"].append(
                        new_examples["attention_mask"]
                    )

                    new_examples = {
                        "input_ids": [],
                        "attention_mask": [],
                    }

                new_examples["input_ids"].extend(exs)
                new_examples["attention_mask"].extend(mask)

            if len(new_examples["input_ids"]) > 0:
                padding_length = block_size - len(new_examples["input_ids"])
                new_examples["input_ids"] = [
                    tokenizer.pad_token_id
                ] * padding_length + new_examples["input_ids"]
                new_examples["attention_mask"] = [0] * padding_length + new_examples[
                    "attention_mask"
                ]

                concatenated_examples["input_ids"].append(new_examples["input_ids"])
                concatenated_examples["attention_mask"].append(
                    new_examples["attention_mask"]
                )

            concatenated_examples["labels"] = concatenated_examples["input_ids"].copy()

            return concatenated_examples

        return {**group_texts(input_tokens)}


def get_trainer(
    model_name: str,
    model: PreTrainedModel | PeftModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: PTT | PTTF,
    seed: int,
    access_token: str,
    no_special_tokens: bool = True,
) -> Trainer:
    """Trainer for the model with various configurations

    Args:
        model_name (str): The name of the model
        model (PreTrainedModel): The model to be trained
        tokenized_splits (Dict[str, Dataset]): The tokenized datasets
        tokenizer (PTT | PTTF): The tokenizer to be used
        seed (int): The seed for reproducibility
        compute_metrics (Callable[[EvalPrediction], Dict[str, float]): The
        function to compute the metrics
        access_token (str): The access token for the model

    Returns:
        Trainer: The trainer for the model
    """

    #  accelerator_config = AcceleratorConfig(split_batches=True)

    no_spec = "NO_SPEC" if no_special_tokens else "SPEC"

    training_args = TrainingArguments(
        output_dir="attention/models/results/" + model_name,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        optim="paged_adamw_8bit",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to=["wandb"],
        seed=seed,
        include_tokens_per_second=True,
        do_eval=True,
        hub_token=access_token,
        push_to_hub=True,
        hub_private_repo=True,
        hub_model_id=model_name.split("/")[1] + "-peft-" + no_spec,
        gradient_checkpointing=True,
        # deepspeed="deepspeed_config.json",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    return trainer


def inference(
    model: PreTrainedModel | PeftModel,
    tokenized_splits: Dataset,
    tokenizer: PTT | PTTF,
    truncate: Tuple[int, int],
    on_dataset="val",
    batch_size=1,
    save_attention=False,
) -> DataFrame:
    """Runs the model on the dataset (saves the attention if save_attention is True) and returns the results

    Args:
        model (PreTrainedModel): the model to be used
        tokenized_splits (Dataset): the tokenized dataset
        tokenizer (PTT | PTTF): the tokenizer to be used
        truncate (Tuple[int, int]): the maximum length of the input and the total length
        on_dataset (str, optional): the split of the dataset to use. Defaults to "val".
        batch_size (int, optional): the batch size. Defaults to 1.
        save_attention (bool, optional): whether to save the attention weights (~1GB per sample). Defaults to False.

    Returns:
        DataFrame: the results of the inference
    """
    results = []

    if save_attention:
        assert batch_size == 1, "Batch size must be 1 to save attention"

    only_dataset = tokenized_splits[on_dataset].select_columns(
        ["input_ids", "labels", "attention_mask"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        only_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        pin_memory=True,
    )

    with torch.inference_mode():
        index = 0
        for batch in tqdm(dataloader):
            # also stop when seeing ### generated
            output = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                max_length=truncate[1],
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_attentions=save_attention,
                return_dict_in_generate=save_attention,
                renormalize_logits=True,
            )

            if save_attention:
                attentions = output["attentions"]
                output = output["sequences"]

                with open(f"attention/models/results/attention/attention-{index}.pkl", "wb") as f:
                    pkl.dump(attentions, f)

                with open(f"attention/models/results/attention/sequences-{index}.pkl", "wb") as f:
                    pkl.dump(output, f)

            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=False)

            results.append(
                (
                    decoded_output,
                    tokenized_splits[on_dataset]["answer"][index : index + batch_size],
                )
            )

            index += batch_size

    df = DataFrame(results, columns=["output", "target"])

    # split the batched results into separate rows
    df = df.explode(["output", "target"])

    # separate string tokens <question>, <context>,
    # <answer> in three different columns
    df["question"] = tokenized_splits[on_dataset]["question"]
    df["context"] = tokenized_splits[on_dataset]["context"]

    def extract_answer(x):
        eos = tokenizer.eos_token

        eos = re.escape(eos)

        answer_matcher = re.compile(rf"#(?:(?!{eos}).|\n)+")

        x = answer_matcher.search(x).group(0)

        if "### Answer: " not in x:
            return x

        if "### End" not in x:
            return x.split("### Answer: ")[1]

        return x.split("### Answer: ")[1].split("### End")[0]

    df["output"] = df["output"].apply(extract_answer)

    df = df.reset_index(drop=True)

    return df


def get_peft_config(model_name, rl=False, layers=None, version=1):
    """Finds the appropriate PEFT configuration for the model

    Args:
        model_name (str): The name of the model
        rl (bool, optional): comes from the rl script. Defaults to False.
        layers (_type_, optional): Restriction on the layers to apply it to. Defaults to None.
        version (int, optional): Sometimes different configs can be used (for gemma). Defaults to 1.

    Raises:
        ValueError: if the model is not supported

    Returns:
        LoraConfig: The PEFT configuration
    """
    r_lora = 32

    # special case in which we learn the output layer too
    if "CobraMamba/mamba-gpt" in model_name:
        target_modules = [
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
        r_lora = 64
    elif "microsoft/phi" in model_name:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "dense",
            "fc1",
            "fc2",
            "lm_head",
        ]
        r_lora = 128
    elif "google/gemma" in model_name:
        if version == 1:
            target_modules = [
                "k_proj",
                "q_proj",  # "o_proj",
                "v_proj",  # "gate_proj", "up_proj",
                "lm_head",  # ,"down_proj"
            ]
            r_lora = 64
        elif version == 2:
            target_modules = [
                "k_proj",
                "q_proj",
                "o_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ]
            r_lora = 64
        else:
            raise ValueError("Version not supported")
    elif "Qwen/Qwen2" in model_name:
        target_modules = [
            "k_proj",
            "q_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
        r_lora = 64
    else:
        raise ValueError("Model not supported")

    if rl:
        #  remove lm_head
        target_modules = [x for x in target_modules if x != "lm_head"]

    # Prepare model for PEFT
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r_lora,
        lora_alpha=r_lora,
        lora_dropout=0.1,
        use_rslora=True,
        target_modules=target_modules,
        bias="none",
        layers_to_transform=layers,
    )

    return peft_config


def get_device_map(
    model_name: str, max_memory={0: "2GiB", 1: "10GiB", "cpu": "15GiB"}
) -> Dict[str, str | int | torch.device] | str:
    """Gets the device map for the model and assigns specific devices to avoid LoRA issues.

    Args:
        model_name (str): The name of the model
        max_memory (dict, optional): The maximum size per device. Defaults to {0: "2GiB", 1: "10GiB", "cpu": "15GiB"}.

    Returns:
        Dict[str, str | int | torch.device] | str: The device map
    """
    if torch.cuda.device_count() == 0:
        max_memory = {"cpu": "15GiB"}
    elif torch.cuda.device_count() == 1:
        max_memory = {0: "10GiB"}

    config = AutoConfig.from_pretrained(model_name)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    model.tie_weights()

    if "CobraMamba/mamba-gpt" in model_name:
        no_split_module_classes = ["GPT2Model", "GPT2LMHeadModel"]
    elif "microsoft/phi" in model_name:
        no_split_module_classes = ["PhiDecoderLayer", "lm_head"]
    elif "google/gemma" in model_name:
        no_split_module_classes = ["GemmaDecoderLayer", "lm_head"]
    elif "EleutherAI/pythia" in model_name:
        no_split_module_classes = ["GPTNeoXDecoderLayer", "lm_head"]
    elif "Qwen/Qwen2" in model_name:
        no_split_module_classes = ["Qwen2DecoderLayer", "lm_head"]
    else:
        no_split_module_classes = []

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes,
    )

    device_map["lm_head"] = 0
    device_map["model.norm"] = 0
    device_map["model.embed_tokens"] = 0

    print(device_map)

    return device_map


def load_model(
    model_name: str,
    from_checkpoint: str,
    access_token: Optional[str] = "",
    save_attention: bool = False,
    no_quant=False,
    trl=False,
    device_map=None,
) -> Tuple[PTT | PTTF, PreTrainedModel]:
    """
    Loads the quantized model from Huggingface or from a checkpoint

    Args:
        model_name (str): The name of the model
        from_checkpoint (str): The name of the checkpoint
        access_token (str, optional): The access token for the model.
        Defaults to "".
        save_attention (bool, optional): Whether to ensure attention can be
        saved. Defaults to False.
        no_quant (bool, optional): Whether to use quantization. Defaults to False.
        trl (bool, optional): SSpecific TRL folder and model. Defaults to False.
        device_map (Optional[Dict[str, str | int | torch.device]]): The device map. Defaults to None.
        

    Returns:
        Tuple[PTT | PTTF, PreTrainedModel]: The tokenizer and the model
    """

    if no_quant:
        quantization_config = None
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if device_map is None:
        device_map = get_device_map(model_name, max_memory={0: "3.5GiB", 1: "10GiB", "cpu": "15GiB"})

    model_name_complete = model_name

    if not trl:
        if from_checkpoint == "none":
            model_name_complete = model_name
        else:
            model_name_complete = "attention/models/results/" + model_name + "/" + from_checkpoint
    else:
        model_name_complete = (
            "attention/models/trl/results/" + model_name + "/" + from_checkpoint
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_complete,
        cache_dir="attention/models/llm" if not trl else None,
        quantization_config=quantization_config,
        token=access_token,
        device_map=device_map,
        attn_implementation="eager" if save_attention else None,
        use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_complete,
        padding_side="left",
        token=access_token,
    )

    return tokenizer, model


def get_prepared_model(
    model_name: str,
    from_checkpoint: str = "",
    access_token: str | None = "",
    train=False,
    save_attention=False,
    loading_checkpoint=False,
    no_special_tokens=False,
    no_quant=False,
    trl=False,
    device_map=None,
) -> Tuple[PTT | PTTF, PreTrainedModel | PeftModel]:
    """Loads the model and prepares it for training or inference

    Args:
        model_name (str): The name of the model
        from_checkpoint (str, optional): The name of the checkpoint to load. Defaults to "".
        access_token (str | None, optional): The HuggingFace read token. Defaults to "".
        train (bool, optional): Whether to prepare for training. Defaults to False.
        save_attention (bool, optional): Ensures attention can be saved. Defaults to False.
        loading_checkpoint (bool, optional): if loading from a checkpoint. Defaults to False.
        no_special_tokens (bool, optional): if not adding special tokens around citations. Defaults to False.
        no_quant (bool, optional): Whether to avoid quantization. Defaults to False.
        trl (bool, optional): Specific TRL folder and model. Defaults to False.
        device_map (_type_, optional): The device map. Defaults to None.

    Returns:
        Tuple[PTT | PTTF, PreTrainedModel | PeftModel]: The tokenizer and the model
    """

    tokenizer, model = load_model(
        model_name=model_name,
        from_checkpoint=from_checkpoint,
        access_token=access_token,
        save_attention=save_attention,
        no_quant=no_quant,
        trl=trl,
        device_map=device_map,
    )

    important_tokens = {
        "mask_token": "<mask>",
        "pad_token": "<pad>",
        "eos_token": "<eos>",
        "bos_token": "<bos>",
        "unk_token": "<unk>",
    }

    if not no_special_tokens:
        # Add special tokens
        special_tokens = {
            "additional_special_tokens": list(SPECIAL_TOKENS.values()),
        }

        tokenizer.add_special_tokens(special_tokens | important_tokens)
    else:
        # Remove special tokens
        tokenizer.add_special_tokens(important_tokens)


    # Always necessary because we sometimes add pad
    if model.config.vocab_size != len(tokenizer):
        print(
            "Resizing token embeddings: from",
            model.config.vocab_size,
            "to",
            len(tokenizer),
        )
        model.resize_token_embeddings(len(tokenizer))

    if train:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        if not loading_checkpoint:
            peft_config = get_peft_config(model_name)

            model = get_peft_model(model, peft_config)

        return tokenizer, model

    if not trl:
        model_path = "attention/models/results/" + model_name + "/" + from_checkpoint
    else:
        model_path = "attention/models/trl/results/" + model_name + "/" + from_checkpoint

    if loading_checkpoint:
        model = PeftModel.from_pretrained(
            model,
            model_path,
            adapter_name="lora",
            is_trainable=False,
        )

    return tokenizer, model
