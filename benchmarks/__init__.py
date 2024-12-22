from .base import Bench
from .ddxplus import create_ddxplus, create_ddxplus_private
from .text_to_sql import create_bird, create_bird_private

classes = locals()

TASKS = {
    "classification_public": create_ddxplus(),
    "sql_generation_public": create_bird(),
    "classification_private": create_ddxplus_private(),
    "sql_generation_private": create_bird_private()
}

def load_benchmark(benchmark_name) -> Bench:
    if benchmark_name in TASKS:
        return TASKS[benchmark_name]
    if benchmark_name in classes:
        return classes[benchmark_name]

    raise ValueError("Benchmark %s not found" % benchmark_name)
