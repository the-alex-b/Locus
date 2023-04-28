# Locus
Locus is a local, simple, append-only, in-memory vector database based on hnswlib.

## Installation
``` bash
pip install locusdb
```
## Example Code
Some example code to illustrate Locus' functionality.

``` python
import numpy as np
from locusdb import Config, Vector, Index

# create a new configuration
config = Config(max_elements=1000, ef_construction=200, M=16, dim=128, space="cosine", storage_location="index.db")

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

print(f"Matches: {results}")
print(f"Items in index: {index.count}")

# store the index on disk
index._store_on_disk()

# load the index from disk
new_index = Index.from_file(config.storage_location)
```

