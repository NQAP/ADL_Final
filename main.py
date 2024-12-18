import gc
import random
import re
import warnings
from abc import ABC, abstractmethod
from typing import override

import faiss
import numpy
import torch
from accelerate import Accelerator
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers import logging as transformers_logging

from base import Agent
from execution_pipeline import main
from utils import RetrieveOrder, strip_all_lines

# Ignore warning messages from transformers
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()


class RAG:
    def __init__(self, rag_config: dict) -> None:
        self.embed_model = AutoModel.from_pretrained(
            rag_config["embedding_model"],
            **rag_config["embedding_model_kwargs"],
            trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            rag_config["embedding_model"], trust_remote_code=True
        )

        self.index = None
        self.id2evidence = {}
        self.embed_dim = len(self.encode_data("Test embedding size"))
        self.insert_acc = 0

        self.seed = rag_config["seed"]
        self.top_k = rag_config["top_k"]
        orders = {member.value for member in RetrieveOrder}
        assert rag_config["order"] in orders
        self.retrieve_order = rag_config["order"]
        random.seed(self.seed)

        self.create_faiss_index()
        # TODO: make a file to save the inserted rows

    def create_faiss_index(self):
        self.index = faiss.IndexFlatL2(self.embed_dim)

    def encode_data(self, sentence: str) -> numpy.ndarray:
        # Tokenize the sentence
        encoded_input = self.tokenizer(
            [sentence], padding=True, truncation=True, return_tensors="pt"
        ).to(self.embed_model.device)
        # Compute token embeddings
        with torch.inference_mode():
            model_output = self.embed_model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0].cpu().numpy()
        feature = sentence_embeddings[0]
        norm = numpy.linalg.norm(feature)
        return feature / norm

    def insert(self, key: str, value: str) -> None:
        """Use the key text as the embedding for future retrieval of the value text."""
        embedding = self.encode_data(key).astype("float32")  # Ensure the data type is float32
        self.index.add(numpy.expand_dims(embedding, axis=0))
        self.id2evidence[str(self.insert_acc)] = value
        self.insert_acc += 1

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Retrieve top-k text chunks"""
        embedding = self.encode_data(query).astype("float32")  # Ensure the data type is float32
        top_k = min(top_k, self.insert_acc)
        distances, indices = self.index.search(numpy.expand_dims(embedding, axis=0), top_k)
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        results = [
            {"link": str(idx), "_score": {"faiss": dist}}
            for dist, idx in zip(distances, indices, strict=False)
        ]
        # Re-order the sequence based on self.retrieve_order
        if self.retrieve_order == RetrieveOrder.SIMILAR_AT_BOTTOM.value:
            results = list(reversed(results))
        elif self.retrieve_order == RetrieveOrder.RANDOM.value:
            random.shuffle(results)

        text_list = [self.id2evidence[result["link"]] for result in results]
        return text_list


class LocalModelAgent(Agent, ABC):
    """
    A base agent that uses multiple models for text generation tasks.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the local model.
        """

        super().__init__(config)
        self.llm_config = config

        self.accelerator = Accelerator(dynamo_backend=self.llm_config["dynamo_backend"])

        self.current_model_index = 0
        self.models = {}
        self.tokenizers = {}

        if not self.llm_config["save_memory"] or len(self.llm_config["model_names"]) == 1:
            for i, _ in enumerate(self.llm_config["model_names"]):
                self.prepare_model(i)

        self.inputs = []
        self.self_outputs = []
        self.rag = RAG(self.llm_config["rag"])

    def prepare_model(self, index: int) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        if self.llm_config["save_memory"]:
            for model in self.models.values():
                self.accelerator.free_memory(model)
                del model

            self.models.clear()
            self.tokenizers.clear()
            torch.cuda.empty_cache()
            gc.collect()

        model_name = self.llm_config["model_names"][index]

        self.models[model_name] = self.accelerator.prepare(
            AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="cuda:0",
                trust_remote_code=True,
            )
        )
        self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    def generate_response(self, messages: list[dict[str, str]]) -> str:
        """
        Generate a response using the local model.
        """

        if self.llm_config["save_memory"] and len(self.llm_config["model_names"]) > 1:
            self.prepare_model(self.current_model_index)
            assert len(self.models) == 1

        model_name = self.llm_config["model_names"][self.current_model_index]
        current_model, current_tokenizer = self.models[model_name], self.tokenizers[model_name]

        text_chat: str = current_tokenizer.apply_chat_template(  # type: ignore
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = current_tokenizer([text_chat], return_tensors="pt").to(current_model.device)

        with torch.inference_mode():
            generated_ids = current_model.generate(
                **model_inputs, max_new_tokens=self.llm_config["max_tokens"], do_sample=False,
                num_beams=3,
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids, strict=False)
        ]

        response = current_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.current_model_index = (self.current_model_index + 1) % len(
            self.llm_config["model_names"]
        )

        self.update_log_info(
            log_data={
                "num_input_tokens": len(current_tokenizer.encode(text_chat)),
                "num_output_tokens": len(current_tokenizer.encode(response)),
            }
        )

        if self.llm_config["save_memory"]:
            del generated_ids
            torch.cuda.empty_cache()
            gc.collect()

        return response

    @override
    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own prediction
        at the current time step.

        Args:
            correctness (bool): Whether the prediction is correct.

        Returns:
            bool: Whether the prediction is correct.
        """

        if correctness:
            question = self.inputs[-1]
            answer = self.self_outputs[-1]
            chunk = self.get_shot_template().format(question=question, answer=answer)
            self.rag.insert(key=question, value=chunk)

        return correctness

    @abstractmethod
    def get_shot_template() -> str:
        pass


class ClassificationAgent(LocalModelAgent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """

    @staticmethod
    def get_system_prompt() -> str:
        return strip_all_lines(
            """\
            Act as a professional medical doctor that can diagnose the patient
            based on the patient profile. Provide your diagnosis in the following
            format: <number>. <diagnosis>
            """.strip()
        )

    @staticmethod
    def get_zeroshot_prompt(option_text: str, text: str) -> str:
        return strip_all_lines(
            f"""
            Act as a medical doctor and diagnose the patient based on the following patient profile:

            {text}

            All possible diagnoses for you to choose from are as follows (one diagnosis per line,
            in the format of <number>. <diagnosis>):

            {option_text}

            Now, directly provide the diagnosis for the patient in the following format:
            <number>. <diagnosis>
            """.strip()
        )

    @staticmethod
    def get_fewshot_template(
        option_text: str,
        text: str,
    ) -> str:
        return strip_all_lines(
            f"""\
            Act as a medical doctor and diagnose the patient based on the provided patient profile.

            All possible diagnoses for you to choose from are as follows (one diagnosis per line,
            in the format of <number>. <diagnosis>):
            {option_text}

            Here are some example cases.

            {{fewshot_text}}

            Now it's your turn.

            {text}

            Now provide the diagnosis for the patient in the following format: <number>. <diagnosis>
            """.strip()
        )

    @staticmethod
    def extract_label(pred_text: str, label2desc: dict[int, str]) -> str:
        numbers = re.findall(pattern=r"(\d+)\.", string=pred_text)

        if len(numbers) == 1:
            number = numbers[0]
            if int(number) in label2desc:
                prediction = number
            else:
                print(f"Prediction {pred_text} not found in the label set. Randomly select one.")
                prediction = random.choice(list(label2desc.keys()))
        elif len(numbers) > 1:
            print(f"Extracted numbers {numbers} is not exactly one. Select the first one.")
            prediction = numbers[0]
        else:
            print(f"Prediction {pred_text} has no extracted numbers. Randomly select one.")
            prediction = random.choice(list(label2desc.keys()))

        return str(prediction)

    @staticmethod
    def get_shot_template() -> str:
        prompt = """
        {question}
        Diagnosis: {answer}
        """.strip()

        return strip_all_lines(prompt)

    @override
    def __call__(self, label2desc: dict[int, str], text: str) -> str:
        """
        Classify the text into one of the labels.

        Args:
            label2desc (dict[str, str]): A dictionary mapping each label to its description.
            text (str): The text to classify.

        Returns:
            str: The label (should be a key in label2desc) that the text is classified into.

        For example:

        ```
        label2desc = {
            "apple": "A fruit that is typically red, green, or yellow.",
            "banana": "A long curved fruit that grows in clusters and has soft pulpy flesh and "
            "yellow skin when ripe.",
            "cherry": "A small, round stone fruit that is typically bright or dark red.",
        }
        text = "The fruit is red and about the size of a tennis ball."
        label = "apple"  # (should be a key in label2desc, i.e., ["apple", "banana", "cherry"])
        ```
        """

        option_text = "\n".join([f"{k!s}. {v}" for k, v in label2desc.items()])
        system_prompt = self.get_system_prompt()
        prompt_zeroshot = self.get_zeroshot_prompt(option_text, text)
        prompt_fewshot = self.get_fewshot_template(option_text, text)

        shots = (
            self.rag.retrieve(query=text, top_k=self.rag.top_k) if (self.rag.insert_acc > 0) else []
        )

        if len(shots):
            fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
            prompt = prompt_fewshot.format(fewshot_text=fewshot_text)
        else:
            print("No RAG shots found. Using zeroshot prompt.")
            prompt = prompt_zeroshot

        messages = [
            {"role": "user", "content": f"{system_prompt}\n{prompt}"},
        ]
        response = self.generate_response(messages)
        prediction = self.extract_label(response, label2desc)

        self.update_log_info(
            log_data={
                "input_pred": messages[0]["content"],
                "output_pred": response,
                "num_shots": str(len(shots)),
            }
        )
        self.inputs.append(text)
        self.self_outputs.append(f"{prediction!s}. {label2desc[int(prediction)]}")

        return prediction


class SQLGenerationAgent(LocalModelAgent):
    """
    An agent that generates SQL code based on the given table schema and the user query.
    """

    @staticmethod
    def get_system_prompt() -> str:
        return strip_all_lines(
            """
            Act as a professional programmer.

            You will be given a table schema and a user query, and you need to generate
            the correct SQL code to answer the user query in the following format:

            ```sql
            <your_SQL_code>
            ```
            """.strip()
        )

    @staticmethod
    def get_zeroshot_prompt(table_schema: str, user_query: str) -> str:
        return strip_all_lines(
            f"""
            {table_schema}

            -- Using valid SQLite, answer the following question for the tables provided above.
            -- Question: {user_query}

            Now, generate the correct SQL code directly in the following format:

            ```sql
            <your_SQL_code>
            ```
            """.strip()
        )

    @staticmethod
    def get_shot_template() -> str:
        return strip_all_lines(
            """
            Question: {question}
            Answer:

            {answer}
            """.strip()
        )

    @staticmethod
    def get_fewshot_template(table_schema: str, user_query: str) -> str:
        return strip_all_lines(
            f"""
            You are performing the text-to-SQL task. Here are some examples:

            {{fewshot_text}}

            Now it's your turn.

            * SQL schema: {table_schema}
            * Using valid SQLite, answer the following question for the SQL schema provided above.
            * Question: {user_query}

            Now, generate the correct SQL code directly in the following format:

            ```sql
            <your_SQL_code>
            ```
            """.strip()
        )

    def __call__(self, table_schema: str, user_query: str) -> str:
        """
        Generate SQL code based on the given table schema and the user query.

        Args:
            table_schema (str): The table schema.
            user_query (str): The user query.

        Returns:
            str: The SQL code that the LLM generates.
        """

        self.reset_log_info()
        system_prompt = self.get_system_prompt()
        prompt_zeroshot = self.get_zeroshot_prompt(table_schema, user_query)
        prompt_fewshot = self.get_fewshot_template(table_schema, user_query)

        shots = (
            self.rag.retrieve(query=user_query, top_k=self.rag.top_k)
            if (self.rag.insert_acc > 0)
            else []
        )
        if len(shots):
            fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
            prompt = prompt_fewshot.format(fewshot_text=fewshot_text)
        else:
            print("No RAG shots found. Using zeroshot prompt.")
            prompt = prompt_zeroshot

        if self.llm_config["model_names"][self.current_model_index] == "google/gemma-2-9b-it":
            messages = [
                {"role": "user", "content": f"{system_prompt}\n{prompt}"},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        pred_text = self.generate_response(messages)
        sql_code = self.parse_sql(pred_text)

        self.update_log_info(
            log_data={
                "num_shots": str(len(shots)),
                "input_pred": prompt,
                "output_pred": pred_text,
            }
        )

        self.inputs.append(user_query)
        self.self_outputs.append(f"```sql\n{sql_code}\n```")
        return sql_code

    @staticmethod
    def parse_sql(pred_text: str) -> str:
        """
        Parse the SQL code from the LLM's response.
        """

        pattern = r"```sql([\s\S]*?)```"
        match = re.search(pattern, pred_text)
        if match:
            sql_code = match.group(1)
            sql_code = sql_code.strip()
            return sql_code
        else:
            print("No SQL code found in the response")
            sql_code = pred_text
        return sql_code


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--bench_name", type=str, required=True)
    args = parser.parse_args()

    if args.bench_name.startswith("classification"):
        agent_name = ClassificationAgent
        max_tokens = 32
        model_names = ["google/gemma-2-9b-it"]
        rag_embedding_model = "dunzhang/stella_en_400M_v5"
        top_k = 5
    elif args.bench_name.startswith("sql_generation"):
        agent_name = SQLGenerationAgent
        max_tokens = 512
        model_names = ["Qwen/Qwen2.5-7B-Instruct"]
        rag_embedding_model = "BAAI/bge-base-en-v1.5"
        top_k = 16
    else:
        msg = f"Invalid benchmark name: {args.bench_name}"
        raise ValueError(msg)

    exp_name = f"{'self' if len(model_names) == 1 else 'mam'}_streamicl_{args.bench_name}_nf4"

    config = {
        "save_memory": True,
        "dynamo_backend": "tensorrt",
        "exp_name": exp_name,
        "bench_name": args.bench_name,
        "model_names": model_names,
        "max_tokens": max_tokens,
        "rag": {
            "embedding_model": rag_embedding_model,
            "seed": 0,
            "top_k": top_k,
            "order": "similar_at_top",
            "embedding_model_kwargs": {},
        },
    }

    bench_cfg = {
        "bench_name": args.bench_name,
        "output_path": f"{args.bench_name}/{exp_name}.csv",
    }

    if config["rag"]["embedding_model"] == "dunzhang/stella_en_400M_v5":
        config["rag"]["embedding_model_kwargs"] = {
            "use_memory_efficient_attention": False,
            "unpad_inputs": False,
        }

    if config["dynamo_backend"] == "tensorrt":
        import torch_tensorrt  # noqa: F401

    agent = agent_name(config)
    main(agent, bench_cfg, use_wandb=True, wandb_name=exp_name, wandb_config=config)
