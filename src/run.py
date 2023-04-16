from locus.index import Index, Vector
import numpy as np
from pprint import pprint

index = Index(dimensions=10)

embedding = np.float32(np.random.random((1, 10)))
structured_data = {"a": 1}
vector = Vector(embedding=embedding, data=structured_data)


index.add_vector(vector)
