



class LeafFamily:
    """
    visualize a hierarchical biological taxonomical tree of plant species
    """
    def __init__(self, name: str):
        self.name = name
        self.children = []


class LeafSpecies(LeafFamily):
    """
    visualize a hierarchical biological taxonomical tree of plant species
    """
    def __init__(self, name: str):
        self.name = name
        self.children = []



class TaxonomyTree(LeafSpecies, LeafFamily):
    """
    visualize the full graph of a hierarchical biological taxonomical tree of plant clades
    """
    def __init__(self, name: str, ancestors: list, children: list):
        self.name = name
        self.children = children
        self.ancestors = ancestors



