import random
import re
import warnings
from abc import ABC, abstractmethod
from typing import override

import faiss
import numpy
import torch
from accelerate import Accelerator
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
        )
        # Compute token embeddings
        with torch.inference_mode():
            model_output = self.embed_model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        feature = sentence_embeddings.numpy()[0]
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
    A base agent that uses a local model for text generation tasks.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the local model.
        """

        super().__init__(config)
        self.llm_config = config

        self.accelerator = Accelerator(dynamo_backend=self.llm_config["dynamo_backend"])

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_config["model_name"],
            quantization_config=bnb_config,
            device_map="cuda:0",
        )
        self.model.eval()
        self.model = self.accelerator.prepare(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_config["model_name"])

        self.inputs = []
        self.self_outputs = []
        self.rag = RAG(self.llm_config["rag"])

    def generate_response(self, messages: list[dict[str, str]]) -> str:
        """
        Generate a response using the local model.
        """

        text_chat: str = self.tokenizer.apply_chat_template(  # type: ignore
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=self.llm_config["max_tokens"], do_sample=False
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids, strict=False)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

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
        system_prompt = """
        Act as a professional medical doctor that can diagnose the patient
        based on the patient profile. Provide your diagnosis in the following
        format: <number>. <diagnosis>
        """.strip()

        return strip_all_lines(system_prompt)

    @staticmethod
    def get_zeroshot_prompt(option_text: str, text: str) -> str:
        prompt = f"""
        Act as a medical doctor and diagnose the patient based on the following patient profile:

        {text}

        All possible diagnoses for you to choose from are as follows (one diagnosis per line,
        in the format of <number>. <diagnosis>):

        {option_text}

        Now, directly provide the diagnosis for the patient in the following format:
        <number>. <diagnosis>
        """.strip()

        return strip_all_lines(prompt)

    @staticmethod
    def get_fewshot_template(
        option_text: str,
        text: str,
    ) -> str:
        prompt = f"""
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

        return strip_all_lines(prompt)

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
            try:
                prompt = re.sub(
                    pattern=r"\{fewshot_text\}", repl=fewshot_text, string=prompt_fewshot
                )
            except Exception as e:
                print(f"Error ```{e}``` caused by these shots. Using the zero-shot prompt.")
                prompt = prompt_zeroshot
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


class SQLGenerationAgent(Agent):
    """
    An agent that generates SQL code based on the given table schema and the user query.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        # TODO
        raise NotImplementedError

    def __call__(self, table_schema: str, user_query: str) -> str:
        """
        Generate SQL code based on the given table schema and the user query.

        Args:
            table_schema (str): The table schema.
            user_query (str): The user query.

        Returns:
            str: The SQL code that the LLM generates.
        """
        # TODO: Note that your output should be a valid SQL code only.
        raise NotImplementedError

    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own SQL code at the current time step.
        """
        # TODO
        raise NotImplementedError


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--bench_name", type=str, required=True)
    args = parser.parse_args()

    if args.bench_name.startswith("classification"):
        agent_name = ClassificationAgent
        max_tokens = 32
    elif args.bench_name.startswith("sql_generation"):
        agent_name = SQLGenerationAgent
        max_tokens = 512
    else:
        msg = f"Invalid benchmark name: {args.bench_name}"
        raise ValueError(msg)

    model_name = "google/gemma-2-9b-it"

    config = {
        "dynamo_backend": "inductor",
        "exp_name": f"self_streamicl_{args.bench_name}_{model_name}_8bit",
        "bench_name": args.bench_name,
        "model_name": model_name,
        "max_tokens": max_tokens,
        "rag": {
            "embedding_model": "dunzhang/stella_en_400M_v5",
            "seed": 0,
            "top_k": 5,
            "order": "similar_at_top",
            "embedding_model_kwargs": {
                "use_memory_efficient_attention": False,
                "unpad_inputs": False,
            },
        },
    }
    bench_cfg = {"bench_name": args.bench_name, "output_path": f"{args.bench_name}.csv"}

    agent = agent_name(config)
    main(agent, bench_cfg, use_wandb=True, wandb_name=config["exp_name"], wandb_config=config)
