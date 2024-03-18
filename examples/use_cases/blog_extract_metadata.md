Advanced Use Case: Metadata Enrichment

There are cases where It can be useful to enrich a Document with metadata before indexing it on a DocumentStore. This can 
be done by filling the metadata field in the Document or by updating an existing field. The metadata can then act as a 
filter later on when retrieving documents from the DocumentStore.

The metadata can be known beforehand, or it can be extracted from the Document's content. In this example, we enrich 
the metadata of a Document with information extracted from the Document's text itself. We use an LLM to extract this 
information and add it to the Document's metadata.

In this example our goal is to extract the number of inhabitants from Wikipedia articles on cities and add it to the 
Document's metadata.

We start by creating a component that instructs an LLM to extract the number of inhabitants from a given Wikipedia article
in a one-shot in-context learning manner. This component is based on the PromptBuilder together with the OpenAIGenerator.

```python
from haystack import component, Pipeline, Document
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder

class LLMMetadataExtractor:

    def __init__(self):
        answer_prompt = """
        Given a Wikipedia article about a city you extract the number of inhabitants or the estimated population 
        living in that city. Usually this information is found in the first.
        It can be an exact number or an estimate. Please always convert the number to an ordinal number format and don't
        include the word inhabitants, population, or any other word that refers to the number of people living in the city
        in the answer. 
        If in the article there is no information about the number of inhabitants, please return None 
        ### Example 1        
        Wikipedia:
        '''
        Berlin (/bɜːrˈlɪn/, bur-LIN; German: [bɛʁˈliːn] ⓘ)[10] is the capital and largest city of Germany by both area 
        and by population.[11] Its more than 3.85 million inhabitants[12] make it the European Union's most populous city, 
        according to population within city limits.[4] Simultaneously, the city is one of the  States of Germany. Berlin 
        is surrounded by the State of Brandenburg and Brandenburg's capital Potsdam is nearby. Berlin's urban area has 
        a population of around 4.5 million and is therefore the most populous urban area in Germany.        
        '''
        Inhabitants: 3.850.000                        
        ### Example 2:
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

    @component.output_types(answer=Document)
    def run(self, document: Document):
        result = self.pipeline.run({'builder': {'document': document.content}})
        inhabitants = result['llm']['replies'][0].strip()
        document.meta["inhabitants"] = inhabitants
        return {"answer": document}
```

We can run this component on a single document to see the result. We will use the Wikipedia python package to get the
content of the Wikipedia article on Lisbon and then run the component on it. We should see the number of inhabitants

```bash
pip install wikipedia
```

Then we initialize the component, get the text for the Wikipedia article on Lisbon, and run the component on it. 

```python
metadata_extractor = LLMMetadataExtractor()
page = wikipedia.page(title='Lisbon', auto_suggest=False)
lisbon_wiki_text = page.content 
lisbon_doc = Document(content=lisbon_wiki_text.split("\n\n")[0])
reply = metadata_extractor.run(lisbon_doc)
reply['answers'].meta
```
We should see the following output:
```bash
>> {'inhabitants': 548703}
```


Let's now create a complete pipeline to index a list of documents in a `DocumentStore` making use of the `LLMMetadataExtractor`
component to enrich the documents' metadata.

First, we need to create an auxiliary component `Looper` that will run the `LLMMetadataExtractor` on a list of documents. 
The `Looper` component will be able to run a component, a pipeline, or any function over a list of documents, and return 
a list of  results. This will allow us to run the `LLMMetadataExtractor` on a list of documents in a single pipeline.


```python
from typing import List, Callable
from haystack import component, Pipeline, Document
from haystack.core.component import Component

@component
class Looper:
    def __init__(self, operation: [Component, Pipeline, Callable]):
        self.operation = operation

    @component.output_types(results=List[Document])
    def run(self, documents: List[Document]):
        results = []

        if isinstance(self.operation, Pipeline) or isinstance(self.operation, Component):
            for doc in documents:
                result = self.operation.run(doc)
                results.append(result['answer'])

        elif isinstance(self.operation, Callable):
            for doc in documents:
                results.append(self.operation(doc))

        return {"results": results}
```

Let's get the first paragraph of the Wikipedia articles on Lisbon, Hamburg, Rome, and Madrid and run the `LLMMetadataExtractor`
using the `Looper` component. 

```python
documents = []
for title in ["Lisbon", "Hamburg", "Rome", "Madrid"]:
    page = wikipedia.page(title=title, auto_suggest=False)
    doc = Document(content=page.content.split("\n\n")[0])
    documents.append(doc)

metadata_extractor = LLMMetadataExtractor()
looper = Looper(operation=metadata_extractor)
looper.run(documents)
```

We should see the following output:
```bash
[{'answer': Document(id=b9444672253f9d222df1bf69cce7675efe13858ab16f42c6d9c7d0ea37dfad2c, content: 'Lisbon (; Portuguese: Lisboa [liʒˈβoɐ] ) is the capital and largest city of Portugal, with an estima...', meta: {'inhabitants': '548,703'})},
 {'answer': Document(id=4e20ac9c88f45ed1209cd1480d49bffed0f50f539e653af94c75e0f0606f8837, content: 'Hamburg (German: [ˈhambʊʁk] , locally also [ˈhambʊɪ̯ç] ; Low Saxon: Hamborg [ˈhambɔːç] ), officially...', meta: {'inhabitants': '1.900.000'})},
 {'answer': Document(id=a12fb9ffb2a4e65b4d83cf798efd67eda8a0c68b883555f4ad285cbaa3885b5e, content: 'Rome (Italian and Latin: Roma, Italian: [ˈroːma] ) is the capital city of Italy. It is also the capi...', meta: {'inhabitants': '2,860,009'})},
 {'answer': Document(id=55a55cc869537a2487e063b93d7bcf70add641468565c7f1c2402686b24c08b1, content: 'Madrid ( mə-DRID, Spanish: [maˈðɾið] ) is the capital and most populous city of Spain. The city has ...', meta: {'inhabitants': '3.400.000'})}]
```

Now we can create a pipeline to index the documents in a `InMemoryDocumentStore` and store the documents enriched with the
metadata extracted by the `LLMMetadataExtractor` component.


```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

metadata_extractor = LLMMetadataExtractor()
looper = Looper(operation=metadata_extractor)

document_store = InMemoryDocumentStore()
document_cleaner = DocumentCleaner()
document_embedder = SentenceTransformersDocumentEmbedder(
    model="thenlper/gte-large", 
    meta_fields_to_embed=['inhabitants']
)
document_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component(name="looper", instance=looper)
indexing_pipeline.add_component(name="cleaner", instance=document_cleaner)
indexing_pipeline.add_component("embedder", document_embedder)
indexing_pipeline.add_component("writer", document_writer)
indexing_pipeline.connect("looper.results", "cleaner")
indexing_pipeline.connect("cleaner", "embedder")
indexing_pipeline.connect("embedder", "writer")
```

The defined document_store is an `InMemoryDocumentStore` contains now the documents enriched with the metadata extracted 
by the LLMetadataExtractor component.