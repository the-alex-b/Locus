# from locus import Index, Vector, Config
# import numpy as np
# from pprint import pprint

# index = Index(dimensions=10)

# embedding = np.float32(np.random.random((1, 10)))
# structured_data = {"a": 1}
# vector = Vector(embedding=embedding, data=structured_data)


# index.add_vector(vector)

import numpy as np

from locus import Config, Index, Vector

# create a new configuration
config = Config(
    max_elements=1000,
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
for i in range(10):
    embedding = np.random.randn(config.dim)
    data = {"id": i, "message": f"test message {i}"}
    vector = Vector(embedding=embedding, data=data)
    vectors.append(vector)

# add the vectors to the index
for vector in vectors:
    index.add_vector(vector)

# retrieve the closest vectors to a query embedding
query_embedding = np.random.randn(config.dim)
results = index.retrieve(query_embedding, number_of_results=3)
print(results)

# store the index on disk
index._store_on_disk()

# load the index from disk
new_index = Index.from_file(config.storage_location)
