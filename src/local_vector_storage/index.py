from __future__ import annotations
import hnswlib
from dataclasses import dataclass
import numpy as np
import pickle


@dataclass
class Config:
    max_elements: int = 100
    ef_construction: int = 200
    M: int = 16
    dim: int = 1536
    space: str = "cosine"
    storage_location: str = "index.db"


class Vector:
    def __init__(self, embedding, data):
        self.embedding: np.array = embedding
        self.data: dict = data


class Index:
    def __init__(self, dimensions, config=Config()):
        self.config = config
        # HNSW index
        self.hnsw_index = hnswlib.Index(space=config.space, dim=dimensions)
        self.hnsw_index.init_index(
            max_elements=config.max_elements,
            ef_construction=config.ef_construction,
            M=config.M,
        )
        # Structured data
        self.structured_memory: dict(Vector) = {}

    @classmethod
    def from_file(cls, file="index.db") -> Index:
        with open(file, "rb") as handle:
            return pickle.load(handle)

    def add_vector(self, vector: Vector, persist_on_disk=True) -> None:
        # add to hnsw index
        self.hnsw_index.add_items(vector.embedding, len(self.structured_memory))

        # add to stuctured data
        self.structured_memory[len(self.structured_memory)] = vector.data

        if persist_on_disk:
            self._store_on_disk()

    def retrieve(self, embedding: np.array, number_of_results=3) -> list[dict]:
        labels, distances = self.hnsw_index.knn_query(embedding, k=number_of_results)

        return [self.structured_memory[id].message for id in labels[0]]

    def _store_on_disk(self) -> None:
        with open(
            self.config.storage_location,
            "wb",
        ) as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
