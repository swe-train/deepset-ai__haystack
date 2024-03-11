"""
**Description**

Working with metadata filters is powerful and can boost performance quite a bit. However, this is often annoying for the users to set.

This RAG pipeline should take a regular text query and then, if the query contains attributes that relate to existing metadata fields in the index, add the according metadata filters to the further query execution.

Example:

- Query: What was the revenue of nvidia in 2022?
- ⇒ meta data filters we can extract automatically
    - “company”: “nvidia”
    - “year”: 2022
    - etc
- (all based on an existing index with meta data fields…)

**Flow**

Query → LLM → query + metadata filters→ run retriever →  Generator → answer
"""
import json
from typing import Dict

from haystack import Pipeline, component, Document
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


@component
class LLMMetadataQueryExtractor:

    def __init__(self):
        prompt = """
        You extract information from a given that matches a given list of metadata fields from some index. 
        
        ###
        Example 1:
        Query: "What was the revenue of Nvidia in 2022?"
        Extracted metadata fields: {"company": "nvidia", "year": 2022}           
        ###
        Example 2:
        Query: "What were the most influential publications in 2023 regarding Alzheimers disease?"
        Extracted metadata fields: {"condition": "Alzheimers", "year": 2023}
        ###
        Example 3:
        Query: "{{query}}"
        Extracted metadata fields:
        """
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=PromptBuilder(prompt))
        self.pipeline.add_component(name="llm", instance=OpenAIGenerator(model="gpt-3.5-turbo"))
        self.pipeline.connect("builder", "llm")

    @component.output_types(query=str, filters=Dict[str, str])
    def run(self, query: str):
        result = self.pipeline.run({'builder': {'query': query}})
        metadata = json.loads(result['llm']['replies'][0])

        # this will probably need to be done with specific data structures and in a more sophisticated way
        filters = []
        for key, value in metadata.items():
            field = f"meta.{key}"
            filters.append({f"field": field, "operator": "==", "value": value})

        return {"query": query, "filters": {"operator": "AND", "conditions": filters}}


def main():

    documents = [
        Document(
            content="top publications of Erick Mayer in 2012",
            meta={"year": 2012, "topics": ["Alzheimers"], "author": "Erick Mayer"}),
    ]
    document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
    document_store.write_documents(documents=documents, policy=DuplicatePolicy.OVERWRITE)

    pipeline = Pipeline()
    metadata_extractor = LLMMetadataQueryExtractor()
    retriever = InMemoryBM25Retriever(document_store=document_store)

    pipeline.add_component(instance=metadata_extractor, name="metadata_extractor")
    pipeline.add_component(instance=retriever, name="retriever")
    pipeline.connect("metadata_extractor.query", "retriever.query")
    pipeline.connect("metadata_extractor.filters", "retriever.filters")

    pipeline.run(data={"metadata_extractor": {"query": "What were the top publications of Erick Mayer in 2012?"}})



