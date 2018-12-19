
class ElF():
    """ Class defining the electronic descriptors used by MLCF. ElF stands for ELectronic Fingerprint

        Parameters:
        ---
        
        value: dict or np.ndarray
             value of descriptor can either be a complex (dict) or real (np.ndarray) tensor.
        angles: np.ndarray (3)
            angles by which ElF was rotated into local coordinate system (used to rotate forces into same CS).
        basis: dict
            basis for elf representation
        species: str,
            atomic species (element symbol)
        unitcell: np.ndarray (3,3)
            unitcell of the system (used by fold_back_coords during alingment)
    """
    def __init__(self, value, angles, basis, species, unitcell):

        self.value = value
        self.angles = angles
        self.basis = basis
        self.species = species
        self.unitcell = unitcell
