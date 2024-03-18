Advanced Use Case: Extract metadata filters from a query

Using metadata filters can be useful to narrow down the search space when querying a DocumentStore. 

Sometimes, the query itself might contain information that can be used as metadata filter during the querying process. 

In this example, we show how knowing beforehand the metadata fields we can leverage on it to extract
metadata from the query and use it to filter the search space when querying a DocumentStore.

We start by creating a component based on the `PromptBuilder` together with the `OpenAIGenerator` that instructs an LLM 
to extract keywords, phrases, or entities from a given query which can be used as metadata filters.

```python
import json
from typing import Dict, List

from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

@component()
class LLMMetadataQueryExtractor:

    def __init__(self):
        prompt = """
        You are part of an information system that processes users queries.
        Given a user query you extract information from it that matches a given list of metadata fields.
        The information to be extracted from the query must match the semantics associated with the given metadata fields.
        The information that you extracted from the query will then be used as filters to narrow down the search space 
        when querying an index.
        The extracted information in 'Extracted metadata' must be returned in a valid JSON format.                
        ###
        Example 1:
        Query: "What was the revenue of Nvidia in 2022?"
        Metadata fields: {"company", "year"}
        Extracted metadata fields: {"company": "nvidia", "year": 2022}           
        ###
        Example 2:
        Query: "What were the most influential publications in 2023 regarding Alzheimers disease?"
        Metadata fields: {"disease", "year"}
        Extracted metadata fields: {"disease": "Alzheimers", "year": 2023}
        ###
        Example 3:
        Query: "{{query}}"
        Metadata fields: "{{metadata_fields}}"
        Extracted metadata fields:
        """
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=PromptBuilder(prompt))
        self.pipeline.add_component(name="llm", instance=OpenAIGenerator(model="gpt-3.5-turbo"))
        self.pipeline.connect("builder", "llm")

    @component.output_types(query=str, filters=Dict[str, str])
    def run(self, query: str, metadata_fields: List[str]):
        result = self.pipeline.run({'builder': {'query': query, 'metadata_fields': metadata_fields}})
        metadata = json.loads(result['llm']['replies'][0])

        # this will probably need to be done with specific data structures and in a more sophisticated way
        filters = []
        for key, value in metadata.items():
            field = f"meta.{key}"
            filters.append({f"field": field, "operator": "==", "value": value})

        return {"query": query, "filters": {"operator": "AND", "conditions": filters}}
```

We can now first test the `LLMMetadataQueryExtractor` in isolation, passing it a query and a list of metadata fields.

```python
extractor = LLMMetadataQueryExtractor()
query = "What were the most influential publications in 2023 regarding Alzheimers disease?"
metadata_fields = ["disease", "year"]
result = extractor.run(query, metadata_fields)
print(result)
```

This will output the following:
```bash
{'query': 'What were the most influential publications in 2023 regarding Alzheimers disease?', 
 'filters': {'operator': 'AND', 
  'conditions': [
    {'field': 'meta.disease', 'operator': '==', 'value': 'Alzheimers'}, 
    {'field': 'meta.year', 'operator': '==', 'value': 2023}
  ]}
}
```

Notice that the `LLMMetadataQueryExtractor` has extracted the metadata fields from the query and returned them in a
format that can be used as filters passed directly to a `Retriever`. By default, the `LLMMetadataQueryExtractor`
joins the conditions with an `AND` operator.


Now, let's plug the `LLMMetadataQueryExtractor` into a pipeline with a retriever connected with a document store to 
see how it works in practice.

We start by creating a DocumentStore and adding some documents to it.

```python
from haystack import Pipeline, Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


documents = [
    Document(
        content="publication about Alzheimers disease prevention research done over 2023 patients study",        
        meta={"year": 2022, "disease": "Alzheimers", "author": "Michael Butter"}),
    Document(
        content="some text about investigation and treatment of Alzheimers disease",
        meta={"year": 2023, "disease": "Alzheimers", "author": "John Bread"}),
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

query = "publications 2023 Alzheimers disease?"
metadata_fields = ["year", "author", "topics"]

pipeline.run(data={"metadata_extractor": {"query": query, "metadata_fields": metadata_fields}})
```