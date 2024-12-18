import json
import textwrap
from abc import ABC, abstractmethod

from utils import setup_logger


# The base class used for classification and multi-choice questions (MCQs)
class Agent(ABC):
    LOG_KEYS = [  # noqa: RUF012
        "num_inference_call",  # number of inference call to the LLM
        "num_success_call",  # (per-call-level) whether the inference / API call is successful
        "num_input_tokens",  # the total number of input tokens in __call__()
        "num_output_tokens",  # the total number of output tokens in __call__()
    ]
    # Prompt for parsing the outputs for mapping to the label space
    PROMPT_PARSE = textwrap.dedent("""\
Model output: {{model_output}}

Convert the model output into one of the following options (one option per line):
{{options}}

Answer (please only answer with a single option):""")

    def __init__(self, config: dict) -> None:
        self.config = config
        # Setup logging info
        self.exp_name = config["exp_name"] if "exp_name" in config else "baseline"
        self.log_path = f"log/{config['bench_name']}/{self.exp_name}.jsonl"
        self.logger = setup_logger(name="jsonlines_logger", log_file=self.log_path)

        # log information of the current data point
        self.log_info = dict.fromkeys(self.LOG_KEYS, 0)

        # accumulation of self.log_info through time steps
        self.accum_log_info = dict.fromkeys(self.LOG_KEYS, 0)

    @abstractmethod
    def __call__(self, prompt: str, label_set: list[str], **kwargs) -> str:
        """
        Generate response text using the prompt.
        The response should be parsed to a label in the label_set.
        """
        raise NotImplementedError

    def initialize(self, train_rows: list[dict]) -> None:
        """(Optional) Initialize the agent with some training instances.

        The training instances should be a list of dictionaries, each of which should at least
        contain the following keys:
        {"desc": <str>, "input": <str>, "output": <str>, "label_set": <set>}

        "desc": The task description (invariant across each instance, e.g., Based on the premise
        and hypothesis provided, determine the relationship between them. Choose the appropriate
        answer from the following options (one option per line):
        \nentailment\nneutral\ncontradiction)
        "x": The training input
        "y": The verbalized label of this instance
        "label_set": All possible labels
        """
        assert isinstance(train_rows, list)
        keys = ["desc", "x", "y", "label_set"]
        for train_row in train_rows:
            assert isinstance(train_row, dict)
            for key in keys:
                assert key in train_row

    @abstractmethod
    def update(self, has_feedback: bool, **feedbacks) -> bool:
        """Return True if the agent is updated in this time_step."""
        raise NotImplementedError

    def reset_log_info(self) -> None:
        self.log_info = dict.fromkeys(self.LOG_KEYS, 0)

    def update_log_info(self, log_data: dict) -> None:
        for k, v in log_data.items():
            if isinstance(v, str) or isinstance(v, list):
                self.log_info[k] = v
            elif isinstance(v, int):
                self.log_info[k] += v
                if k in self.accum_log_info.keys():
                    self.accum_log_info[k] += v
            else:
                msg = f"error key-value pair: {k} -> {v} ({v} should be either str, int, or list)"
                raise TypeError(msg)

    def get_wandb_log_info(self) -> dict:
        log_data = {}
        for key in self.LOG_KEYS:
            log_data[key] = self.log_info[key]
            log_data[f"total_{key}"] = self.accum_log_info[key]
        return log_data

    def log(self, label_text: str | None = None) -> None:
        """This method should be called at the end of each time_step."""
        self.log_info["label_text"] = label_text
        self.logger.info(json.dumps(self.log_info))
        self.reset_log_info()

    def get_options_text(self, label_set: set[str]) -> str:
        """Convert the label_set into the option text."""
        return "\n".join(list(label_set))
