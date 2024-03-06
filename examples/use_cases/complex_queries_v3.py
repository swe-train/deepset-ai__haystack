import json
from typing import List, Dict

from datasets import load_dataset, Dataset

from haystack import Pipeline, Document, component
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

embedder_model = "sentence-transformers/all-MiniLM-L6-v2"

"""
**Description**

This pipeline should split complex queries into subqueries and then run retrieval for each of the sub-queries before 
feeding data to another LLM to generate an answer. Examples:

1. Query: Did Microsoft or Google make more money last year?
    - Split: How much profit did Microsoft make? // How much profit did Google make?
    - Get answers for each of the sub-queries and then use them to generate an answer.
        - Sub-query: How much profit did Microsoft make?
        - Answer: Microsoft made 3,141,590 dollars
        - Sub-query: How much profit did Google make?
        - Answer: Google made 2,718,280 dollars
    - Merge results into a structure and feed to LLM to generate the answer: 
    {'query': 'Did Microsoft or Google make more money last year?',
      'sub-questions': [{'query': "How much profit did Microsoft make last year?", 'answers': ['3.14159 dollars']},
                        {'query': "How much profit did Google make last year?", 'answers': ['2.71828 dollars']}
                        ]}


**Flow**

Query → LLM → decomposed query → run multiple retrievers → merge results → LLM → answer

1. Decompose complex query: the query is split into multiple sub-queries using an LLM.
2. Run multiple retrievers: Each sub-query is run through a retriever to get the top-k documents.
3. Merge results: The results from the retrievers are merged.
4. LLM: The merged results are fed to an LLM to generate an answer.
"""


def get_doc_stored_indexed_data(data: Dataset):
    document_store = InMemoryDocumentStore()
    pipeline = Pipeline()

    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=10))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=embedder_model))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy="skip"))

    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")
    pipeline.run({"cleaner": {"documents": [Document.from_dict(doc) for doc in data["train"]]}})

    return document_store


@component
class ReplySplitter:

    @component.output_types(queries=List[Document])
    def run(self, replies: List[str]):
        replies = [r.split(' // ') for r in replies]
        return {"queries": [Document(content=subquery) for subquery in replies[0]]}


@component
class LLMReasoningMultipleQueries:

    def __init__(self):
        answer_prompt = """        
        You answer a complex query which was split into multiple sub-questions.
        You inspect the sub-questions answers to generate an answer for the complex question.
        The sub-questions can have multiple answers and you need to merge them or select the best one or discard some.
        The query and the sub-questions are provided as JSON data.
        ###
        Example 1:
        Complex Question: "{'query': 'Did Microsoft or Google make more money last year?',
                           'sub-questions': [
                                {'query': "How much profit did Microsoft make last year?", 'answers': ['3.14159 dollars']},
                                {'query': "How much profit did Google make last year?", 'answers': ['2.71828 dollars']}
                            ]
                           }"
        Answer: Microsoft made more money last year.
        ###
        Example 2:
        Example 1:
        Complex Question: "{'query': 'Who's older Joe Biden or Donald Trump?',
                           'sub-questions': [
                                    {'query': "How old is Joe Biden?", 'answers': ['81 years old']},
                                    {'query': "How old is Donald Trump?", 'answers': ['77 years old']}
                            ]
                           }"
        Answer: Joe Biden is older than Donald Trump.
        ###
        Example 3:
        Complex Question: {{question}}
        Answer:
        """

        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=PromptBuilder(answer_prompt))
        self.pipeline.add_component(name="llm", instance=OpenAIGenerator(model="gpt-3.5-turbo"))
        self.pipeline.connect("builder", "llm")

    @component.output_types(answers=Dict)
    def run(self, collected_answers: Dict):
        data = {"question": json.dumps(collected_answers)}
        results = self.pipeline.run(data=data)
        return {"answer": results['llm'], 'question': collected_answers}


@component
class SubqueriesAnswers:

    def __init__(self, doc_store: InMemoryDocumentStore):
        self.extractive_retriever_pp = self._build_extractive_retriever_pp(doc_store)

    @staticmethod
    def _build_extractive_retriever_pp(document_store):
        retriever = InMemoryEmbeddingRetriever(document_store=document_store)
        reader = ExtractiveReader()
        reader.warm_up()
        extractive_qa_pipeline = Pipeline()
        extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
        extractive_qa_pipeline.add_component(instance=reader, name="reader")
        extractive_qa_pipeline.connect("retriever.documents", "reader.documents")
        return extractive_qa_pipeline

    @component.output_types(queries=Dict)
    def run(self, complex_question: str, queries: List[Document]):

        collected_answers = {
            "query": complex_question,
            "subqueries": []
        }

        for doc in queries:
            results = self.extractive_retriever_pp.run(
                data={"retriever": {'query_embedding': doc.embedding, 'top_k': 3},
                      "reader": {"query": doc.content, "top_k": 2}}
            )
            if results:
                answers = {"query": doc.content, "answers": []}
                for answer in results['reader']['answers']:
                    if answer.data:
                        answers['answers'].append(answer.data)
                collected_answers['subqueries'].append(answers)

        return {"queries": collected_answers}


def main():
    # Load data and index it in the document store, and create an extractive retriever
    data = load_dataset("Tuana/game-of-thrones")
    doc_store = get_doc_stored_indexed_data(data)

    query = "Who has more sibilings, Jaime Lannister or Jonh Snow?"
    query = "Which family has more members, the Lannisters or the Starks?"
    query = "What are the names of the lands that belong to the Lannisters and to the Starks?"
    query = "Who is older, Jaime Lannister or Jonh Snow?"

    splitter_prompt = """
    You are a query engine.
    You prepare queries that will be send to a web search component.
    Sometimes, these queries are very complex.
    You split up complex queries into multiple queries so that you can run multiple searches to find an answer.
    When you split a query, you separate the sub-queries with '//'.
    If the query is simple, then keep it as it is.
    ###
    Example 1:
    Query: Did Microsoft or Google make more money last year?
    Split: How much profit did Microsoft make? // How much profit did Google make?
    ###
    Example 2:
    Query: What is the capital of Germany?
    Split: What is the capital of Germany?
    ###
    Example 3:
    Query: {{question}}
    Split:
    """

    builder = PromptBuilder(splitter_prompt)
    llm = OpenAIGenerator(model="gpt-3.5-turbo")
    splitter = ReplySplitter()
    sentence_embedder = SentenceTransformersDocumentEmbedder(model=embedder_model)
    sentence_embedder.warm_up()

    pipeline = Pipeline()

    # Add components that decouple a complex query into multiple sub-queries
    pipeline.add_component(name="builder", instance=builder)
    pipeline.add_component(name="llm", instance=llm)
    pipeline.add_component(name="splitter", instance=splitter)
    pipeline.add_component(name="embedder", instance=sentence_embedder)
    pipeline.connect("builder", "llm")
    pipeline.connect("llm", "splitter")
    pipeline.connect("splitter", "embedder")

    # pipeline.run(data={"builder": {"question": query}})

    complex_query = SubqueriesAnswers(doc_store)

    # Add components that run retrievers for each sub-query and merge the results
    pipeline.add_component(name="complex_query", instance=complex_query)
    pipeline.connect("embedder", "complex_query")

    # pipeline.run(data={"builder": {"question": query}, "complex_query": {"complex_question": query}})

    # Add components that generate an answer from the merged results
    llm_prompt = LLMReasoningMultipleQueries()
    pipeline.add_component(name="answer", instance=llm_prompt)
    pipeline.connect("complex_query", "answer")

    # Run the pipeline with the complex query
    pipeline.run(data={"builder": {"question": query}, "complex_query": {"complex_question": query}})


if __name__ == '__main__':
    main()

