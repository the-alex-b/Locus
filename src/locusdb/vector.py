import numpy as np


class Vector:
    """A class representing a vector with an embedding and associated data.

    Parameters
    ----------
    embedding (np.array): A numpy array representing the embedding of the vector.
    data (dict): A dictionary containing the data associated with the vector.

    Attributes
    ----------
    embedding (np.array): A numpy array representing the embedding of the vector.
    data (dict): A dictionary containing the data associated with the vector.
    """

    def __init__(self, embedding, data):
        self.embedding: np.array = embedding
        self.data: dict = data
