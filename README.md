# ADL_Final

# prepare environment

`conda env create --prefix ./venv -f environment.yaml`

`conda activate ./venv` : activate this environment

# install requirement

`pip install -r requirements.txt`

# set up the necessary datasets:

`python setup_data.py`

# model_name (change the model_name in main.py) ex.  model_names = ["google/gemma-2-9b-it"]

`Qwen/Qwen2.5-7B-Instruct`
`meta-llama/Llama-3.1-8B-Instruct`
`prince-canuma/Ministral-8B-Instruct-2410-HF`

# run

`python main.py --bench_name "classification_public"`
`python main.py --bench_name "sql_generation_public"`

ADL 2024 final
