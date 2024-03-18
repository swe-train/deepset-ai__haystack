Advanced Use Case: Query Expansion

Query expansion is the process of reformulating a query to improve the quality of the search results. More generally, 
the query expansion adds search terms to a user's search to match documents. The intent is to improve precision and/or recall
with the additional synonyms and related words.

In this example, we will create a custom component the `LLMQueryExpander` to expand a given query into multiple queries 
that are similar in meaning. This custom component uses the `PromptBuilder` to create a prompt and the `OpenAIGenerator`
to generate the expanded queries.

```python
from typing import List
from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator


@component
class LLMQueryExpander:

    def __init__(self):
        answer_prompt = """
        You are part of an information system that processes users queries.    
        You expand a given query into multiple queries that are similar in meaning. 
        Use a structure like the following example to expand the given query into multiple queries that are similar in meaning.
        ###
        Example Query 1: "climate change effects"
        Example Expanded Queries: ["impact of climate change", "consequences of global warming", "effects of environmental changes"]
        ###
        Example Query 2: ""machine learning algorithms""
        Example Expanded Queries: ["neural networks", "clustering", "supervised learning", "deep learning"]
        ###
        Query: "{{query}}"
        Example Expanded Queries:        
        """
        builder = PromptBuilder(answer_prompt)
        llm = OpenAIGenerator(model="gpt-3.5-turbo")
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=builder)
        self.pipeline.add_component(name="llm", instance=llm)
        self.pipeline.connect("builder", "llm")

    @component.output_types(expanded=List[str], query=str)
    def run(self, query: str):
        result = self.pipeline.run({'builder': {'query': query}})
        expanded_query = eval(result['llm']['replies'][0])
        return {"expanded": list(expanded_query), "query": query}
```

Let's test the `LLMQueryExpander` component.

```python
expander = LLMQueryExpander()
expander.run(query="open source nlp frameworks")
```

The output will be:

```bash
{'expanded': ['natural language processing frameworks',
  'free nlp tools',
  'open source language processing software',
  'nlp libraries with open source code']}
```

We now need to issue a query to the document store to retrieve the documents that match the expanded query. We will use 
a `Looper` component to iterate over the expanded queries and issue a query to the document store for each expanded query.

```python
from typing import Callable, List

from haystack import Pipeline, Document, component
from haystack.core.component import Component

@component
class Looper:
    def __init__(self, operation: [Component, Pipeline, Callable]):
        self.operation = operation
        self.results = []
        self.ids = set()

    def add_document(self, document: Document):
        if document.id not in self.ids:
            self.results.append(document)
            self.ids.add(document.id)
        
    @component.output_types(documents=List[Document])
    def run(self, queries: List[str]):
        if isinstance(self.operation, Pipeline) or isinstance(self.operation, Component):
            for query in queries:
                result = self.operation.run(query)
                for doc in result['documents']:
                    self.add_document(doc)

        elif isinstance(self.operation, Callable):
            for query in queries:
                result = self.operation(query)
                for doc in result['documents']:
                    self.add_document(doc)

        return {'documents': self.results}
```

The `Looper` component will be able to run a component, a pipeline, or any function over a list of documents, and return 
a list of results.

This component is used to handle a list of queries. For each query, retrieves documents from a `DocumentStore` and returns 
a list of documents that are unique across all queries.

Let's test the `Looper` component connected to a simple `DocumentStore` together with the `LLMQueryExpander` component.

```python
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

documents = [
    Document(content="The effects of climate are many including loss of biodiversity"),
    Document(content="The impact of climate change is evident in the melting of the polar ice caps."),
    Document(content="Consequences of global warming include the rise in sea levels."),
    Document(content="One of the effects of environmental changes is the change in weather patterns.")
]

doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
doc_store.write_documents(documents)

retriever = InMemoryBM25Retriever(document_store=doc_store, top_k=3)
query_expander = LLMQueryExpander()
query_expansion = query_expander.run(query="climate change")
looper = Looper(operation=retriever)
looper.run(queries=query_expansion['expanded'])
```

Now lets build a complete `Pipeline` connecting the `LLMQueryExpander` and the `Looper` components but also a custom 
component `LLMAnswerQuery` that uses the `PromptBuilder` and `OpenAIGenerator` to summarize and answer to the query using 
the textual content from the documents retrieved by the query expansions made from the original query.


```python
from typing import List

from haystack import Pipeline, component, Document
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
    
@component
class LLMAnswerQuery:

    def __init__(self):
        answer_prompt = """        
        You are part of an information system that summarises related documents.
        You answer a query using the textual content from the documents retrieved by the query expansions made         
        from the original query.
        You build the summary answer based only on quoting information from the documents.
        You should quote the documents you used to support your answer.
        ###
        Original Query: "{{query}}"
        Retrieved Documents: {{documents}}
        Summary Answer:
        """
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=PromptBuilder(answer_prompt))
        self.pipeline.add_component(name="llm", instance=OpenAIGenerator(model="gpt-3.5-turbo"))
        self.pipeline.connect("builder", "llm")

    @component.output_types(answer=str)
    def run(self, query: str, documents: List[Document]):
        results = self.pipeline.run(data={"query": query, "documents": documents})
        return {"answer": results}
```

We first need to create a `DocumentStore` and populate it with some documents. We will use the `wikipedia` library to
retrieve the first sentence of some Wikipedia pages, and then index them into as documents in the `DocumentStore`.

```python
import wikipedia
from haystack import Pipeline, Document
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

def get_doc_store():
    raw_docs = []
    wikipedia_page_titles = ["Electric_vehicle", "Dam", "Electric_battery", "Tree", "Solar_panel", "Nuclear_power",
                             "Wind_power", "Hydroelectricity", "Coal", "Natural_gas", "Greenhouse_gas", "Renewable_energy", 
                             "Fossil_fuel"]
    for title in wikipedia_page_titles:
        page = wikipedia.page(title=title, auto_suggest=False)
        first_sentence = page.content.split("\n\n")[0]
        doc = Document(content=first_sentence, meta={"title": page.title, "url": page.url})
        raw_docs.append(doc)

    doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("cleaner", DocumentCleaner())    
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.SKIP))
    indexing_pipeline.connect("cleaner", "writer")    
    indexing_pipeline.run({"cleaner": {"documents": raw_docs}})

    return doc_store
```

Now we can create a `Pipeline` that connects the `LLMQueryExpander`, the `Looper`, and the `LLMAnswerQuery` components.


```python
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack import Pipeline

doc_store = get_doc_store()

retriever = InMemoryBM25Retriever(document_store=doc_store, top_k=3)
looper = Looper(operation=retriever)
query_expander = LLMQueryExpander()

pipeline = Pipeline()
pipeline.add_component(name="expander", instance=query_expander)
pipeline.add_component(name="looper", instance=looper)
pipeline.add_component(name="llm", instance=LLMAnswerQuery())

pipeline.connect("expander.expanded", "looper")
pipeline.connect("looper", "llm.documents")
pipeline.connect("expander.query", "llm.query")

pipeline.run({"query": "green sources of energy"})
```

```bash
{'llm': {'answer': {'llm': {'replies': ['Some green sources of energy include renewable energy, hydroelectricity, and wind power. Renewable energy comes from renewable resources that are natural and replenishable. Hydroelectricity is generated from hydropower, while wind power utilizes wind energy to generate useful work. These sources are considered green as they have lower carbon emissions compared to fossil fuels like coal, oil, and natural gas. (Source: https://en.wikipedia.org/wiki/Renewable_energy, https://en.wikipedia.org/wiki/Hydroelectricity, https://en.wikipedia.org/wiki/Wind_power)'],
    'meta': [{'model': 'gpt-3.5-turbo-0125',
      'index': 0,
      'finish_reason': 'stop',
      'usage': {'completion_tokens': 112,
       'prompt_tokens': 505,
       'total_tokens': 617}}]}}}}
```