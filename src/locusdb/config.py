from dataclasses import dataclass


@dataclass
class Config:
    """
    A simple data class that contains the configuration options for the index.

    Attributes
    ----------
        max_elements (int): The maximum number of elements in the index. Default is 100.
        ef_construction (int): The construction time parameter for the HNSW index. Default is 200.
        M (int): The number of bi-directional links to add per level in the HNSW index. Default is 16.
        dim (int): The dimension of the embedding space. Default is 1536.
        space (str): The distance metric to use for the HNSW index. Default is "cosine".
        storage_location (str): The file location to use for storing the index on disk. Default is "index.db".
    """

    max_elements: int = 100
    ef_construction: int = 200
    M: int = 16
    dim: int = 1536
    space: str = "cosine"
    storage_location: str = "index.db"
