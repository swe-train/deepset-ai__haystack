import os
from numpy import array, mean
from typing import List

from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack import component, Document
from haystack.components.converters import OutputAdapter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator

# We need to ensure we have the OpenAI API key in our environment variables
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_KEY'

# Initializing standard Haystack components
generator = OpenAIGenerator(
    model="gpt-3.5-turbo",
    generation_kwargs={"n": 5, "temperature": 0.75, "max_tokens": 400},
)

generator = OllamaGenerator(
    model="orca-mini",
    url="http://localhost:11434/api/generate"
)

prompt_builder = PromptBuilder(
    template="""Given a question, generate a paragraph of text that answers the question.    Question: {{question}}    Paragraph:""")

adapter = OutputAdapter(
    template="{{answers | build_doc}}",
    output_type=List[Document],
    custom_filters={"build_doc": lambda data: [Document(content=d) for d in data]}
)

embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
embedder.warm_up()


# Adding one custom component that returns one, "average" embedding from multiple (hypothetical) document embeddings
@component
class HypotheticalDocumentEmbedder:

    @component.output_types(hypothetical_embedding=List[float])
    def run(self, documents: List[Document]):
        stacked_embeddings = array([doc.embedding for doc in documents])
        avg_embeddings = mean(stacked_embeddings, axis=0)
        hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
        return {"hypothetical_embedding": hyde_vector[0].tolist()}