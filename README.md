# Environment set-up
```bash
conda create -n context pandas numpy scikit-learn nltk ipykernel jupyter pyarrow peft transformers optimum evaluate scikit-learn wandb absl-py sympy sentence-transformers tqdm seaborn matplotlib-base python-dotenv gpustat accelerate pytorch pytorch-cuda=12.1 trl -c pytorch -c nvidia -c conda-forge
pip install bitsandbytes
pip install rouge_score
pip install bert_score
```

# Dataset

The annotated dataset is situated under processing/annotated_datasets and can be loaded with any of the class in processing/citation_dataset.

If you want to reproduce this step you can download the databricks-dolly-15k dataset into processing/raw_datasets/databricks-dolly-15k, run processing/dataset_construction (expected to crash halfway through), annotate using Doccano the two parts under processing/parsed_datasets and save them into annotated_datasets. Finally re-run dataset_construction and it should go all the way through to produce the clear_dataset under processing/annotated_dataset.

# How to

Everything can be ran from the "super_script.py".

Ensure you have a .env with READ_TOKEN, WRITE_TOKEN and WANDB_ENTITY defined. These will be used for downloading models (gemma in particular), push models to hub (for attention SFT training) and read runs to make plots (classifier).

Note: many configurations for the attention methods are made to run on the setup of two Nvidia 2080ti, lower vRAM capacities may require tuning the arguments for max_memory.

## General Arguments

    --script:
            the sub script to run
    --gpu:
            cuda GPU to use, works for [0, 0 1], untested beyond
    --wandb:
            online if true, offline if not set

## Attention

This part focuses on the base, SFT and RL attention methods.

### Arguments

The following arguments are the same for the SFT and RL script:

    --model_name: 
            hf model name
        Default: "google/gemma-2b"

    --from_checkpoint: 
            name of the checkpoint folder
            "" if directly saved under model_name
            "latest" to find the latest checkpoint (must exist >0, doe not work for trl)
        Default: "none" for base model

    --max_length_input:
            number of token in (Question, Context) to truncate the dataset to
    --max_length:
            number of token in (Question, Context, Answer) to truncate the dataset to

### SFT train dataset

There are two additional arguments specific to this script (only remove if certain):

    --train
            otherwise evaluates
    --no_special_token
            otherwise adds token to the tokenizer, model and dataset to surround citations in the answer

```bash
python super_script.py --script learning_tokens --from_checkpoint="none" --model_name="google/gemma-2b" --train --no_special_tokens --gpu 0 1 --max_length_input 2048 --max_length 2500 --wandb
```

### RL trainings

Parameters determined by the analysis of SFT model in next section.
(add --save-first if first time)

```bash
python super_script.py --script rl_training --gpu 0 1 --from_checkpoint="checkpoint-166" --layer 8 --head 6 --reward meteor+attn-pen --train_length 60 --max_length 500 --max_length_input 450 --rank 0 --threshold 0.002 --batch_size 64 --wandb
```

### Analysis

To determine hyper-parameters and to evaluate the methods, we can use the "analyse_attention" sub-script:

```bash
# Base model, no nothing
python super_script.py --script analyse_attention --model_name="google/gemma-2b" --from_checkpoint="none" --gpu 0 --max_length_input 2048 --max_length 2500 --split "val" --no_special_tokens
# SFT
python super_script.py --script analyse_attention --model_name="google/gemma-2b" --from_checkpoint="checkpoint-166" --gpu 0 --max_length_input 2048 --max_length 2500 --split "val" --no_special_tokens
# After Reinforcement
python super_script.py --script analyse_attention --model_name="google/gemma-2b" --from_checkpoint="trained_checkpoint-166_meteor+attn-pen_L8_H6_S42_M500_MI450_V1_T60_1" --gpu 0 --max_length_input 2048 --max_length 2500 --split "val" --no_special_tokens --trl
```

```bash
python super_script.py --script analyse_attention_results --iou_path "attention/models/results/google/gemma-2b/attention/iou-gemma-2b-CKPnone-val-total-S42.csv"
python super_script.py --script analyse_attention_results --iou_path "attention/models/results/google/gemma-2b/attention/iou-gemma-2b-CKPcheckpoint-166-val-total-S42.csv"
python super_script.py --script analyse_attention_results --iou_path "attention/models/results/google/gemma-2b/attention/iou-gemma-2b-CKPtrained_checkpoint-166_meteor+attn-pen_L8_H6_S42_M500_MI450_V1_T60_1-val-total-S42.csv"
```

## Baseline

Use for evaluating and fine-tuning various models using different metrics and datasets. The script supports methods like N-grams and embedding-based approaches (e.g., BERT, Nomic, SFR) for citation prediction tasks.

### Arguments

- `--ft`: Fine-tune the model. If this flag is set, the script will run the fine-tuning process.
- `--test`: Evaluate the model. If this flag is set, the script will run the evaluation process.
- `--method`: Specifies the method to use. Options are `ngramsTopK`, `embeddingTopK`, `embeddingThreshold`, `ngramsThreshold`. Default is `ngramsTopK`.
- `--model_name`: Name of the embedding model to use. Default is `bert`. Options are `bert`, `nomic`, `sfr`.
- `--threshold`: Threshold value for evaluation in threshold-based methods. Required for `embeddingThreshold` and `ngramsThreshold` methods.
- `--ft_step`: Number of steps for fine-tuning. Default is `90`.
- `--ft_start`: start value of the threshold for fine-tuning. Default is `0.1`
- `--ft_end`: end value of the threshold for fine-tuning. Default is `0.9`
- `--k`: List of top-K values to evaluate. Default is `[1, 2, 3, 4, 5]`.
- `--batch_size`: Batch size for evaluation. Default is `4`.
- `--plot`: If set, the script will generate plots for the evaluation results. Default is `False`.

### Example Commands

To fine-tune the bert model using embedding threshold method:

```bash
python super_script.py --script baseline.py --ft --method embeddingThreshold --model_name bert --ft_step 90 --gpu 0
```

To fine-tune the sfr model using embedding threshold method:

```bash
python super_script.py --script baseline.py --ft --method embeddingThreshold --model_name sfr --ft_step 90
```

To fine-tune the N-grams threshold method:

```bash
python super_script.py --script baseline.py --ft --method ngramsThreshold --ft_step 90
```

To evaluate the N-grams top-K method:

```bash
python super_script.py --script baseline.py --test --method ngramsTopK --k 1 2 3 4 5 --plot
```
To evaluate the embedding top-K method:

```bash
python super_script.py --script baseline.py --test --method embeddingTopK --model_name sfr --k 1 2 3 4 5 --plot --batch_size=2
```

To evaluate a model using embedding threshold method:

```bash
python super_script.py --script baseline.py --test --method embeddingThreshold --model_name bert --threshold 0.5 
```

Note: When using the embedding method with a large model like SFR, it is recommended to use a small batch size, such as 2 or 1, to avoid exceeding the VRAM.

### Main Functions

- `main()`: This function orchestrates the splitting of the dataset, distribution analysis, and evaluation of the models using N-grams and embedding methods.

- `evaluate()`: This function is used specifically for evaluating the models without fine-tuning.

### Supporting Functions

- `compute_metrics(dataset, printed=False, title='')`: Computes the Intersection over Union (IoU) metrics for the dataset.
  
- `print_metrics(dataset, title, type="sub_questions")`: Prints the computed metrics in a formatted table.

- `plot_IoU(series, title="IoU_box_plot")`: Plots the IoU scores as a box plot.

- `split_dataframe(df, frac_val=0.1, frac_eval=0.1, seed=42)`: Splits the dataframe into training, validation, and evaluation sets.

- `fine_tune_threshold(dataset, model="ngrams", threshold_space=np.linspace(0.1, 0.9, 80), batch_size=4)`: Fine-tunes the threshold for a given model using the dataset.

- `evaluate_threshold(dataset, threshold, model="ngrams", batch_size=4)`: Evaluates the performance of the model on the dataset using a specific threshold.

- `evaluate_embedding(dataset, model_name, k=1, title='', batch_size=4)`: Evaluates the performance of embedding models on the dataset.

- `evaluate_TopK_embedding(dataset, model_name, k=[1, 2, 3, 4, 5], plot=False, batch_size=4)`: Evaluates the top-K performance of embedding models and optionally plots the results.

- `evaluate_NGrams(dataset, k, title='')`: Evaluates the performance of N-grams models on the dataset.

- `evaluate_TopK_NGrams(dataset, k=[1, 2, 3, 4, 5], plot=False)`: Evaluates the top-K performance of N-grams models and optionally plots the results.

- `dataset_distribution(dataset, type="sub_questions", title="Dataset distribution")`: Analyzes and plots the distribution of the dataset.

### Notes

- Ensure the dataset path is correct in the `main()` and `evaluate()` functions.
- Modify the `Embedder` initialization in `fine_tune_threshold` and `evaluate_embedding` functions based on your specific model needs.
- The plotting functions save plots to a `plots` directory. Ensure this directory exists or create it before running the script.

## Classifier

### Command Line Arguments

You can run the script from the command line with the following arguments:
- `--train`: Train the model. If this flag is set, the script will run the training process. 
- `--test`: Evaluate the model. If this flag is set, the script will run the evaluation process.
- `--model_name`: The name of the pre-trained model to use (e.g., `google/gemma-2b`, `mistralai/Mistral-7B-v0.1`, `distilbert-base-uncased`, `roberta-base`).
- `--run_name`: A name for the training run.
- `--lr`: Learning rate for the optimizer. Default is `5e-5`.
- `--lora_drop`: Dropout rate for the LoRA layers. Default is `0.1`.
- `--lora_r`: Dimension of the low-rank matrices in LoRA. Default is `4`.
- `--lora_alpha`: The alpha parameter of the LoRA. Default is `4`.
- `--dataset_input`: the input types for classification (e.g, `A&Q`, `S3&A&Q`, `S3&A`, `A`). Default is `A&Q`.
- `--augment`: Whether to use data augmentation.
- `--batch_size`: Batch size for training and evaluation. Default is `8`.
- `--train_size`: Fraction of the dataset to use for training. Default is `1.0`.

### Example

Generic example:
```bash
python super_script.py --script llm_classifier.py --model_name "distilbert-base-uncased" --run_name "my_experiment" --lr 5e-5 --lora_drop 0.1 --lora_r 1 --augment 0 --batch_size 8 --train_size 1.0
```
Roberta-Base best model training:
```bash
python super_script.py --script llm__classifier.py --model_name=roberta-base --train --dataset_input="S3&A&Q"
```
Distilbert best model training and testing:
```bash
python super_script.py --script llm__classifier.py --model_name=distilbert-base-uncased  --train --test --dataset_input="S3&A&Q"
```
Gemma-2b best model training and testing:
```bash
python super_script.py --script llm__classifier.py --model_name=google/gemma-2b --train --test --batch_size=3 --dataset_input="S3&A&Q"
```

### Class: `CC_classifier`

- **`__init__(self, model_name, dataset_input="A&Q", model_checkpoint=None, tokenizer_checkpoint=None)`**:
  Initializes the classifier with the specified model and input type.

- **`train(self, run_name, lr=5e-5, lora_drop=0.1, lora_r=1, augment=0, batch_size=8, train_size=1.0)`**:
  Trains the model with the specified parameters.

- **`make_prediction(self, answer, question, context, load_bar=None)`**:
  Makes predictions on the context using the trained model.

### Functions

- **`split_dataframe(df, frac_val=0.1, frac_eval=0.1, seed=42)`**:
  Splits the dataframe into training, validation, and evaluation sets.

### Training Configuration

The training is configured using the `TrainingArguments` from the `transformers` library, and a `Trainer` instance is used to manage the training loop.

### Input Types

The script supports various input types for classification:

- **A**: Answer only
- **A&Q**: Answer and Question
- **S3&A**: Sentence context of three sentences and Answer
- **S3&A&Q**: Sentence context of three sentences, Answer, and Question

### Example Code

Here is a sample code snippet to illustrate how the classifier is initialized and used:

```python
# Initialize the classifier
classifier = CC_classifier(model_name="distilbert-base-uncased")

# Train the classifier
classifier.train(run_name="my_experiment")

# Make a prediction
predictions = classifier.make_prediction(answer="This is the answer.", question="What is the question?", context="This is the context sentence.")
print(predictions)
```

## Saving and Loading Models

The trained model and tokenizer are saved in the `llm_classifier/train_model/model/` and `llm_classifier/train_model/tokenizer/` directories respectively. You can load these models for future predictions.

### Saving Example

```python
trainer.model.save_pretrained("llm_classifier/train_model/model/my_experiment")
self.tokenizer.save_pretrained("llm_classifier/train_model/tokenizer/my_experiment")
```

## Notes

- Ensure that your environment has the necessary libraries installed.
- Use the appropriate model names supported by the `transformers` library.
- The script assumes that the dataset is in a specific format, make sure your data adheres to it.

# Reference

Soon to be published

For more details, refer to the comments and structure within the `LLM_classifier.py` script.
