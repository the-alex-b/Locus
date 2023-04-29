import numpy as np
from locusdb import Config, Index, Vector
import cProfile


def profile_performance():
    num_of_elements = 1000

    # create a new configuration
    config = Config(
        max_elements=num_of_elements,
        ef_construction=200,
        M=16,
        dim=128,
        space="cosine",
        storage_location="index.db",
    )

    # create a new index instance
    index = Index(dimensions=config.dim, config=config)

    # create some random vectors
    vectors = []
    for i in range(num_of_elements):
        embedding = np.random.randn(config.dim)
        data = {"id": i, "message": f"test message {i}"}
        vector = Vector(embedding=embedding, data=data)
        vectors.append(vector)

    # add the vectors to the index
    for vector in vectors:
        index.add_vector(vector, persist_on_disk=False)  # persisting is very expensive

    # retrieve the closest vectors to a query embedding
    query_embedding = np.random.randn(config.dim)
    results = index.retrieve(query_embedding, number_of_results=3)

    print(f"Matches: {results}")
    print(f"Items in index: {index.count}")

    # store the index on disk
    index.persist_on_disk()

    # load the index from disk
    new_index = Index.from_file(config.storage_location)

    # retrieve the closest vectors to a query embedding
    # query_embedding = np.random.randn(config.dim)
    # results = new_index.retrieve(query_embedding, number_of_results=3)
    # print(results)


cProfile.run("profile_performance()")
