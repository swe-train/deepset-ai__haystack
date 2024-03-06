# This pipeline uses an LLM to extract information from documents in an indexing pipeline and adds it to the document’s metadata.

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
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from typing import List
from dotenv import load_dotenv


def main():

    load_dotenv()

    prompt_template = """
    You extract information on the number of inhabitants from Wikipedia articles on cities.
    ###
    Example 1:
    Wikipedia:
    '''
    Berlin (/bɜːrˈlɪn/, bur-LIN; German: [bɛʁˈliːn] ⓘ)[10] is the capital and largest city of Germany
    by both area and by population.[11] Its more than 3.85 million inhabitants[12] make it the European
    Union's most populous city, according to population within city limits.[4] Simultaneously, the city is one of the States of Germany. Berlin is surrounded by the State of Brandenburg and Brandenburg's capital Potsdam is nearby. Berlin's urban area has a population of around 4.5 million and is therefore the most populous urban area in Germany.[5][13] The Berlin-Brandenburg capital region has around 6.2 million inhabitants and is Germany's second-largest metropolitan region after the Rhine-Ruhr region and the sixth biggest metropolitan region by GDP in the European Union.[14]
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

    builder = PromptBuilder(prompt_template)
    llm = OpenAIGenerator(model="gpt-3.5-turbo")

    pp = Pipeline()
    pp.add_component(name="builder", instance=builder)
    pp.add_component(name="llm", instance=llm)

    pp.connect("builder", "llm")

    documents = [
        Document(content="With an estimated population in 2022 of 8,335,897 distributed over 300.46 square miles (778.2 km2),[4] the city is the most densely populated major city in the United States. New York has more than double the population of Los Angeles, the nation's second-most populous city.[19] New York is the geographical and demographic center of both the Northeast megalopolis and the New York metropolitan area, the largest metropolitan area in the U.S. by both population and urban area. With more than 20.1 million people in its metropolitan statistical area[20] and 23.5 million in its combined statistical area as of 2020, New York City is one of the world's most populous megacities.[21] The city and its metropolitan area are the premier gateway for legal immigration to the United States. As many as 800 languages are spoken in New York,[22] making it the most linguistically diverse city in the world. In 2021, the city was home to nearly 3.1 million residents born outside the U.S.,[19] the largest foreign-born population of any city in the world.[23]"),
        Document(content="The City of Paris is the centre of the Île-de-France region, or Paris Region, with an official estimated population of 12,271,794 inhabitants on 1 January 2023, or about 19% of the population of France,[2] The Paris Region had a GDP of €765 billion (US$1.064 trillion, PPP)[8] in 2021, the highest in the European Union.[9] According to the Economist Intelligence Unit Worldwide Cost of Living Survey, in 2022, Paris was the city with the ninth-highest cost of living in the world.[10]")
    ]

    pp.run(data={"documents": documents}) # wrong, I know


