from collections.abc import Sequence
from rdkit import Chem



class mol2_supplier():
    def __init__(self, path, **kwargs):
        self.path = path
        self._kwargs = kwargs

    def iter_file(self):
        block = []
        with open(self.path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                if block and line.startswith("@<TRIPOS>MOLECULE"):
                    yield block
                    block = []
                block.append(line)
            yield block


    def get_molecules(self, rdkit=True):
        if rdkit == False:
            blocks = [mol for mol in self.iter_file()]
        else:
            blocks = [block_to_mol(mol) for mol in self.iter_file()]

        return blocks
    

    def __len__(self):
        with open(self.path, "r") as f:
            n_mols = sum(line.startswith("@<TRIPOS>MOLECULE") for line in f)
        return n_mols


def block_to_mol(block):
    mol = Chem.MolFromMol2Block("".join(block), sanitize=False, removeHs=False)
    return mol
