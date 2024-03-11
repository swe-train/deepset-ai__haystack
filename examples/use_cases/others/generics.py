from abc import ABC, abstractmethod
from typing import Dict, Any

from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators import AzureOpenAIGenerator, HuggingFaceTGIGenerator, OpenAIGenerator


@component
class LLMGenerator:
    def __init__(self, prompt: str, generator: BaseGenerator):
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="prompt", instance=PromptBuilder(prompt))
        self.pipeline.add_component(name="llm", instance=generator)
        self.pipeline.connect("prompt", "llm")

    @component.output_types(answers=Any)
    def run(self, prompt_params: str):
        self.pipeline.run(data={"prompt": prompt_params})


def test():
    prompt = ("You summarize documents into bullet points in a way that is easy to understand. "
              "You summarize the following document: {{ document }}. Summary:")

    llm = LLMGenerator(prompt=prompt, generator=AzureOpenAIGenerator())
    llm = LLMGenerator(prompt=prompt, generator=HuggingFaceTGIGenerator())
    llm = LLMGenerator(prompt=prompt, generator=OpenAIGenerator())

    document = "The quick brown fox jumps over the lazy dog."

    llm.run(data={"llm": {"document": document}})
