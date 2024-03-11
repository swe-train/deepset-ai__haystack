"""
**Description**

This pipeline should paraphrase an input query and “expand” this one query into multiple ones. This is particularly
interesting for pipelines with a BM25 retriever (+ a reranker). Might perform better than a vector search in many
use cases.

**Flow**

Query → LLM → multiple queries → run multiple retrievers → merge results → LLM → answer

(don’t see any challenges for 2.0 here
when we get [Query Decomposition](https://www.notion.so/Query-Decomposition-b271b120c2724b189281940a8c20c02e?pvs=21)
working. Very similar use cases, but we should showcase both in our docs)
"""
from typing import Callable, List, Dict

from wikipedia import wikipedia

from haystack import Pipeline, Document, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.core.component import Component
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


@component
class LLMQueryExpander:

    def __init__(self):
        answer_prompt = """        
        You expand a given query into multiple queries that are similar in meaning. Use a structure like the following
        example to expand the given query into multiple queries that are similar in meaning.
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

    @component.output_types(expanded=List[str])
    def run(self, query: str):
        result = self.pipeline.run({'builder': {'query': query}})
        expanded_query = eval(result['llm']['replies'][0])
        print(f"Expanded query: {query}: {expanded_query}")
        return {"expanded": list(expanded_query)}


@component
class Looper:
    def __init__(self, operation: [Component, Pipeline, Callable]):
        self.operation = operation

    @component.output_types(documents=List[Document])
    def run(self, queries: List[str]):
        results = []
        if isinstance(self.operation, Pipeline) or isinstance(self.operation, Component):
            for query in queries:
                result = self.operation.run(query)
                results.extend([doc for doc in result['documents']])

        elif isinstance(self.operation, Callable):
            for doc in queries:
                results.append(self.operation(doc))

        return {'documents': results}


@component
class MergeResults:
    @component.output_types(query=str, context=str)
    def run(self, query: str, documents: List[Document]):
        return {
            'query': query,
            'context': '\n'.join([doc.content for doc in documents])
        }


@component
class LLMAnswerQuery:

    def __init__(self):
        answer_prompt = """        
        You answer a query with a paragraph of text, using as context the text from documents retrieved with expansions 
        from the original query. Use only the text from the context documents to answer the query.
        ###
        Original Query: "{{query}}"
        Context: {{documents}}        
        """
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=PromptBuilder(answer_prompt))
        self.pipeline.add_component(name="llm", instance=OpenAIGenerator(model="gpt-3.5-turbo"))
        self.pipeline.connect("builder", "llm")

    @component.output_types(answers=Dict)
    def run(self, query: str, context: str):
        results = self.pipeline.run(data={"query": query, "documents": context})
        return {"answer": results}


def get_doc_store():
    raw_docs = []
    wikipedia_page_titles = ["Electric_vehicle", "Dam", "Electric_battery", "Tree", "Solar_panel", "Nuclear_power",
                             "Wind_power", "Hydroelectricity", "Coal", "Natural_gas", "Beatles", "Guns_N'_Roses",
                             "Climate_change", "Global_warming", "Greenhouse_gas", "Renewable_energy",]
    for title in wikipedia_page_titles:
        page = wikipedia.page(title=title, auto_suggest=False)
        first_sentence = page.content.split(".")[0]
        doc = Document(content=first_sentence, meta={"title": page.title, "url": page.url})
        raw_docs.append(doc)

    doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("cleaner", DocumentCleaner())
    indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=2))
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.SKIP))

    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "writer")
    indexing_pipeline.run({"cleaner": {"documents": raw_docs}})

    return doc_store


def main():

    doc_store = get_doc_store()

    retriever = InMemoryBM25Retriever(document_store=doc_store, top_k=3)
    query_expander = LLMQueryExpander()
    looper = Looper(operation=retriever)

    pipeline = Pipeline()
    pipeline.add_component(name="expander", instance=query_expander)
    pipeline.add_component(name="looper", instance=looper)
    pipeline.add_component(name="merger", instance=MergeResults())
    pipeline.add_component(name="llm", instance=LLMAnswerQuery())

    pipeline.connect("expander.expanded", "looper")
    pipeline.connect("looper", "merger")
    pipeline.connect("merger.query", "llm.query")
    pipeline.connect("merger.context", "llm.context")

    pipeline.run({"query": "measures climate change"})
