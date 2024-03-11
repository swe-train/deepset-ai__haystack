from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy


def create_indexing_pipeline(document_store, metadata_fields_to_embed=None):
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

    return indexing_pipeline


import wikipedia
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

some_bands = """The Beatles,The Cure""".split(",")

raw_docs = []

for title in some_bands:
    page = wikipedia.page(title=title, auto_suggest=False)
    doc = Document(content=page.content, meta={"title": page.title, "url": page.url})
    raw_docs.append(doc)

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
document_store_with_embedded_metadata = InMemoryDocumentStore(embedding_similarity_function="cosine")

indexing_pipeline = create_indexing_pipeline(document_store=document_store)
indexing_with_metadata_pipeline = create_indexing_pipeline(
    document_store=document_store_with_embedded_metadata, metadata_fields_to_embed=["title"]
)

indexing_pipeline.run({"cleaner": {"documents": raw_docs}})
indexing_with_metadata_pipeline.run({"cleaner": {"documents": raw_docs}})
