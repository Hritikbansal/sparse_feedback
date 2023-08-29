# Sparse Feedback

## Installation

```pip install -r requirements.txt```

## Large-scale Feedback Data from GPT-3.5-Turbo

You can access the large-scale ratings and rankings feedback data at 🤗 [hf datasets](https://huggingface.co/datasets/hbXNov/sparse_feedback)

## Feedback Acquistion 

### Collect Instruction Data

1. The first step would be to collect instructions data in the `instructions.json` format. 
2. The file should be formatted as:
```
[
{"instruction": instruction_1, "input": input_1 (Optional as many instances may not have any input)}, 
{"instruction": instruction_2, "input": input_2}, 
.
.
]
```
3. In our work, we merge three instructions datasets: a subset of [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/main/databricks-dolly-15k.jsonl), [UserOrient Self Instruct](https://github.com/yizhongw/self-instruct/blob/main/human_eval/user_oriented_instructions.jsonl), and a subset of [SuperNI-V2](https://github.com/allenai/natural-instructions). Feel free to create your custom instructions dataset. Our subset of SuperNI-V2 is present [here](data/instructions/).

### Response Generation

1. Once the `data.json` is ready, we use Alpaca-7B model to generate five responses for each instance.
2. Use Alpaca [original repo](https://github.com/tatsu-lab/stanford_alpaca) to setup the environment and download the model checkpoints.
3. Use this script [inference_alpaca](scripts/inference_alpaca.py) to generate CSV file with instructions, input, response1, response2, response3, response4, response5 in it.
4. Sample Alpaca generation is present [here](data/alpaca_generation/sample_generation.json).

### Prompting LLMs for feedback

#### Ratings

1. Convert the `json` file to a `csv` file using the conversion code [here](utils/convert_data_to_csv_ratings.py)
```python
python utils/convert_data_to_csv_ratings.py --input data/alpaca_generation/sample_generation.json --output data/alpaca_generation/ratings_sample_generation.csv
``` 
2. Run the [llm_feedback_ratings.py](scripts/llm_feedback_ratings.py) using the following sample command:
```python
OPENAI_API_KEY=<Your OAI API KEY> python llm_feedback_ratings.py --input_csv ../data/alpaca_generation/ratings_sample_generation.csv --save_feedback_csv ../data/alpaca_generation/feedback_ratings_sample_generation.csv 
```

#### Rankings
1. Convert the `json` file to a `csv` file using the conversion code [here](utils/convert_data_to_csv_rankings.py)
```python
python utils/convert_data_to_csv_rankings.py --input data/alpaca_generation/sample_generation.json --output data/alpaca_generation/rankings_sample_generation.csv 
```
2. Run the [llm_feedback_rankings.py](scripts/llm_feedback_rankings.py) using the following sample command:
```python
 OPENAI_API_KEY=<Your OAI API KEY> python scripts/llm_feedback_rankings.py --input_csv  data/alpaca_generation/rankings_sample_generation.csv --save_feedback_csv data/alpaca_generation/feedback_rankings_sample_generation.csv 
```

#### Consistency

Run the [consistency.py](scripts/consistency.py) with the following command:

```python
python consistency.py --ratings_csv <ratings csv w/ AI feedback> --rankings_csv <rankings csv w/ AI feedback>
```

### Reward Modeling

We use Alpaca-7B as the backbone for the ratings and rankings reward models. please setup Alpaca checkpoints before moving forward. We use single A6000 GPU to train our reward models. Since, we use HF trainer, it should be straightforward to extend the code to multi-GPUs.

Some of the code is adopted from [Alpaca-LoRA](https://github.com/tloen/alpaca-lora), thanks to its developers!

#### Ratings Reward Model

1. Split your ratings feedback data into train.csv and val.csv
2. Setup wandb dashboard for logging purposes.
3. Sample command to get started with the reward model:
```python
 CUDA_VISIBLE_DEVICES=4 python train.py  --output_dir <dir> --train_input_csv <train.csv> --test_input_csv <val.csv> --model_name <path to alpaca 7b>
```
4. You should be able to see the trained checkpoints being stored in your `output_dir` after every epoch.

#### Rankings Reward Model

1. Firstly, we convert the rankings feedback data into a suitable format for reward model training. Run [convert_data_for_ranking_rm.py](utils/convert_data_for_ranking_rm.py). 
```python
 python convert_data_for_ranking_rm.py --input ../data/alpaca_generation/feedback_rankings_sample_generation.csv --output ../data/alpaca_generation/feedback_rankings_for_rm_sample.json
```
2. The output of the above code would be:
```
[
    'i_1': {'sentences': [[a1, a2], [a3, a4]...]}
    ...
    'i_n': {'sentences': [[b1, b2], [b3, b4]...]}
]
``` 
where `[a1, a2]` and `[a3, a4]` are pair of responses for the instruction `i1` such that `a1` is preferred over `a2` and `a3` is preferred over `a4`.

3. Split the created json into `train.json` and `val.json`. Sample command to launch the training:
```python
CUDA_VISIBLE_DEVICES=4 python train.py --per_device_train_batch_size 1 --output_dir <output_dir> --train_input <train.json> --test_input <val.json> --model_name <path to alpaca 7b> --learning_rate 1e-4 --run_name test --gradient_accumulation_steps 4 
```
4. You should be able to see the trained checkpoints being stored in your `output_dir` after every epoch.

### Best-of-64 Policy (Re-Ranking)

1. We utilize [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval/blob/main/example/outputs.json) (thanks to the developers!) as the outputs of the reference model `text-davinci-003` on unseen instructions data.
2. Firstly, generate `64` responses for the `553` unseen instructions from Alpaca-7B model. Run this command to do so:
```python
python inference/eval_inference_alpaca.py --device_id 3 --input_data data/inference_model_outputs/davinci003.json --save_generations data/inference_model_outputs/alpaca7b.json --model_path <path to alpaca-7b>
```
This code should generate a file that looks like [alpaca.json](data/inference_model_outputs/alpaca7b_temp1_64.json)
3. We will re-rank the 64 responses from the model using the reward models trained in the previous step. Run this command to do so:
```python
python inference/reranking.py --device_id 3 --input_data data/inference_model_outputs/alpaca7b_temp1_64.json --save_generations data/inference_model_outputs/ratings_alpaca.json --reward_model_path <path to checkpoint-xxx> --alpaca_model_path <path to alpaca 7B> --reward_model_name ratings_alpaca
```

### Evaluation

#### Rating-based Evaluation

1. Create a `csv` files from the `json` of responses using the following command
```python
python utils/convert_data_for_ratings_eval.py --input data/inference_model_outputs/davinci003.json --output data/inference_model_outputs/davinci_for_llm_eval.csv
python utils/convert_data_for_ratings_eval.py --input data/inference_model_outputs/ratings_alpaca.json --output data/inference_model_outputs/ratings_for_llm_eval.csv
```
2. Run the following command to rate the outputs using LLM:
```python
OPENAI_API_KEY=<YOUR OPENAI KEY> python scripts/llm_feedback_ratings.py --input_csv data/inference_model_outputs/davinci_for_llm_eval.csv --save_feedback_csv data/inference_model_outputs/feedback_ratings_davinci_for_llm_eval.csv 
OPENAI_API_KEY=<YOUR OPENAI KEY> python scripts/llm_feedback_ratings.py --input_csv data/inference_model_outputs/ratings_for_llm_eval.csv --save_feedback_csv data/inference_model_outputs/feedback_ratings_ratings_for_llm_eval.csv 
```
3. Run the following command to get the win-rate for the aligned LLM against the reference model:
```python
python scripts/calc_win_rate_from_ratings_eval.py --input1 data/inference_model_outputs/feedback_ratings_davinci_for_llm_eval.csv --input2 data/inference_model_outputs/feedback_ratings_ratings_for_llm_eval.csv 
```

#### Ranking-based Evaluation

1. Create a single `csv` file from the `json` files containing the model outputs -- reference and aligned LLM.
```python
python utils/convert_data_for_rankings_eval.py --davinci_file data/inference_model_outputs/davinci003.json --alpaca_file data/inference_model_outputs/ratings_alpaca.json --output data/inference_model_outputs/rankings_for_llm_eval.csv
```
2. Run the following command to get the rankings for the pairwise judgments from the LLM
```python
 OPENAI_API_KEY=<YOUR OPENAI API KEY> python scripts/llm_feedback_rankings.py --input_csv data/inference_model_outputs/rankings_for_llm_eval.csv --save_feedback_csv data/inference_model_outputs/feedback_rankings_davincivsratings_for_llm_eval.csv

```
3. Run the following command to get the win-rate for the aligned LLM against the reference model:
```python
python scripts/calc_win_rate_from_rankings_eval.py --input data/inference_model_outputs/feedback_rankings_davincivsratings_for_llm_eval.csv 
```

## Human Feedback Data

TODO