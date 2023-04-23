import tempfile
from typing import Dict, List

import numpy as np
import pytest

from locusdb.index import Config, Index, Vector


@pytest.fixture
def index():
    dimensions = 10
    config = Config(space="cosine", max_elements=1000, ef_construction=200, M=16)
    index = Index(dimensions=dimensions, config=config)
    return index


def test_add_and_retrieve_multiple_vectors(index):
    # Add multiple vectors to the index
    embeddings = np.random.rand(5, 10)
    data_list = [{"key": f"value_{i}"} for i in range(5)]
    for i, embedding in enumerate(embeddings):
        vector = Vector(embedding, data_list[i])
        index.add_vector(vector)

    # Retrieve the top 3 closest vectors to a random embedding
    query_embedding = np.random.rand(10)
    retrieved = index.retrieve(query_embedding, number_of_results=3)

    # Assert that the retrieved data is correct
    assert len(retrieved) == 3
    assert all([isinstance(result["distance"], np.float32) for result in retrieved])
    assert all(result["element"] in data_list for result in retrieved)


def test_add_and_retrieve_large_number_of_vectors(index):
    # Add a large number of vectors to the index
    n_vectors = 1000
    embeddings = np.random.rand(n_vectors, 10)
    data_list = [{"key": f"value_{i}"} for i in range(n_vectors)]
    for i, embedding in enumerate(embeddings):
        vector = Vector(embedding, data_list[i])
        index.add_vector(vector)

    # Retrieve the top 3 closest vectors to a random embedding
    query_embedding = np.random.rand(10)
    retrieved = index.retrieve(query_embedding, number_of_results=3)

    # Assert that the retrieved data is correct
    assert len(retrieved) == 3
    assert all(isinstance(result["distance"], np.float32) for result in retrieved)
    assert all(result["element"] in data_list for result in retrieved)


def test_load_from_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create Index with multiple vectors and store on disk
        config = Config(storage_location=tmpdir + "/test_index.db")
        index = Index(dimensions=3, config=config)
        vectors = [
            Vector([0.1, 0.2, 0.3], {"data": "vector1"}),
            Vector([0.4, 0.5, 0.6], {"data": "vector2"}),
            Vector([0.7, 0.8, 0.9], {"data": "vector3"}),
        ]
        for vector in vectors:
            index.add_vector(vector, persist_on_disk=True)

        # Load from disk
        loaded_index = Index.from_file(config.storage_location)

        # Assert that loaded index has the same vectors as the original index
        for vector in vectors:
            results = loaded_index.retrieve(vector.embedding, number_of_results=1)
            assert results[0]["element"] == vector.data
