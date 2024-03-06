from typing import List, Callable

from haystack import component, Document, Pipeline
from haystack.core.component import Component


@component
class Looper:
    def __init__(self, data: List[Document], operation: [Component, Pipeline, Callable]):
        self.input = data
        self.operation = operation

    def run(self):
        results = []

        if isinstance(self.operation, Pipeline) or isinstance(self.operation, Component):
            for doc in self.input:
                results.append(self.operation.run({'data': doc}))

        elif isinstance(self.operation, Callable):
            for doc in self.input:
                results.append(self.operation(doc))

        return results
