from __future__ import annotations
import hnswlib
import numpy as np
import pickle
from locus.config import Config
from locus.vector import Vector


class Index:
    """
    The Index class is used to create an HNSW index to store and retrieve vectors using their embeddings.

    Parameters
    ----------
    dimensions : int
        An integer representing the dimension of the embedding space.
    config : Config, optional
        A Config instance containing configuration options for the index.

    Methods
    -------
    add_vector(vector: Vector, persist_on_disk: bool = True) -> None:
        Adds a new vector to the index with the specified embedding and data. The persist_on_disk parameter can be used to
        control whether the index is stored on disk after adding the vector.
    retrieve(embedding: np.ndarray, number_of_results: int = 3) -> List[Dict]:
        Retrieves the top number_of_results vectors that are closest to the given embedding. The method returns a list
        of dictionaries containing the data associated with each vector.
    from_file(file: str = "index.db") -> Index:
        Creates a new index instance from the specified file location.

    Attributes
    ----------
    config : Config
        A Config instance containing configuration options for the index.
    hnsw_index : hnswlib.Index
        The HNSW index used to store the embeddings.
    structured_memory : Dict[Vector]
        The structured data associated with the embeddings stored in the index.

    Examples
    --------
    >>> config = Config()
    >>> index = Index(dimensions=10, config=config)
    >>> vector = Vector(np.random.rand(10), {'data': 'example'})
    >>> index.add_vector(vector)
    >>> result = index.retrieve(np.random.rand(10))
    """

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

    def retrieve(self, embedding: np.array, number_of_results: int = 3) -> list[dict]:
        labels, distances = self.hnsw_index.knn_query(embedding, k=number_of_results)

        print(labels)
        print(distances)

        return [
            {"element": self.structured_memory[id], "distance": distances[0][i]}
            for i, id in enumerate(labels[0])
        ]

    def _store_on_disk(self) -> None:
        with open(
            self.config.storage_location,
            "wb",
        ) as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
