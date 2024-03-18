# This pipeline uses an LLM to extract information from documents in an indexing pipeline and adds it to the document’s metadata.
from wikipedia import wikipedia

# **Flow**

# Documents → LLM → Extracted values → add values to metadata → embed documents → store documents

# **Code**

# Is it possible to do this in a single Pipeline, without the external for-loop?
#
# **Open questions / issues**
#
# - [ ]  How can we do this task for each document but still use batch processing in the embedder that should come later?
# - [ ]  Can we send multiple prompts to the LLM in a single batch?


from haystack import component, Pipeline, Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from typing import List, Callable

from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.core.component import Component
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


@component
class Looper:
    def __init__(self, operation: [Component, Pipeline, Callable]):
        self.operation = operation

    @component.output_types(results=List[Document])
    def run(self, documents: List[Document]):
        results = []

        if isinstance(self.operation, Pipeline) or isinstance(self.operation, Component):
            for doc in documents:
                results.append(self.operation.run(doc))

        elif isinstance(self.operation, Callable):
            for doc in documents:
                results.append(self.operation(doc))

        return results


class LLMMetadataExtractor:

    def __init__(self):
        answer_prompt = """
        You extract information on the number of inhabitants from Wikipedia articles on cities.
        ###
        Example 1:
        Wikipedia:
        '''
        Berlin (/bɜːrˈlɪn/, bur-LIN; German: [bɛʁˈliːn] ⓘ)[10] is the capital and largest city of Germany
        by both area and by population.[11] Its more than 3.85 million inhabitants[12] make it the European
        Union's most populous city, according to population within city limits.[4] Simultaneously, the city is one of the 
        States of Germany. Berlin is surrounded by the State of Brandenburg and Brandenburg's capital Potsdam is nearby. 
        Berlin's urban area has a population of around 4.5 million and is therefore the most populous urban area in Germany.
        [5][13] The Berlin-Brandenburg capital region has around 6.2 million inhabitants and is Germany's second-largest 
        metropolitan region after the Rhine-Ruhr region and the sixth biggest metropolitan region by GDP in the European 
        Union.[14]
        '''
        Inhabitants: 3500000
        ###
        Example 2:
        Wikipedia:
        '''
        {{document}}
        '''
        Inhabitants:
        """

        builder = PromptBuilder(answer_prompt)
        llm = OpenAIGenerator(model="gpt-3.5-turbo")
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=builder)
        self.pipeline.add_component(name="llm", instance=llm)
        self.pipeline.connect("builder", "llm")

    @component.output_types(answers=Document)
    def run(self, document: Document):
        result = self.pipeline.run({'builder': {'document': document}})
        inhabitants = result['llm']['replies'][0]
        document.meta["inhabitants"] = int(inhabitants) if inhabitants.isdigit() else None
        return {"answers": document}


def main():

    raw_docs = []
    for title in ["Lisbon", "Paris", "Berlin", "Havana"]:
        page = wikipedia.page(title=title, auto_suggest=False)
        doc = Document(content=page.content, meta={"title": page.title, "url": page.url})
        raw_docs.append(doc)

    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="sentence", split_length=2)
    document_embedder = SentenceTransformersDocumentEmbedder(
        model="thenlper/gte-large",
        meta_fields_to_embed=metadata_fields_to_embed,
    )
    document_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("cleaner", document_cleaner)

    indexing_pipeline.add_component("splitter", document_splitter)
    indexing_pipeline.add_component("embedder", document_embedder)
    indexing_pipeline.add_component("writer", document_writer)

    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    metadata_extractor = LLMMetadataExtractor()
    looper = Looper(operation=metadata_extractor)
    looper.run(documents)

    pipeline = Pipeline()
    pipeline.add_component(name="looper", instance=looper)

