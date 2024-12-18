# ADL_Final

ADL 2024 final

## Get started

Install dependencies:

```sh
conda env create --yes --prefix ./venv -f ./environment.yaml
conda activate ./venv
pip install -r requirements.txt
```

## set up the necessary datasets

```sh
python setup_data.py
```

## choose model

In our project, we have tested the following models:\\
Qwen/Qwen2.5-7B-Instruct\\
meta-llama/Llama-3.1-8B-Instruct\\
prince-canuma/Ministral-8B-Instruct-2410-HF\\
google/gemma-2-9b-it

If you want to use your own model, change the model_name in main.py ex.  model_names = ["google/gemma-2-9b-it"]

## run

If you want to run classification part, then run:

```sh
python main.py --bench_name "classification_public"
```

If you want to run text-to-sql part, then run:

```sh
python main.py --bench_name "sql_generation_public"
```
