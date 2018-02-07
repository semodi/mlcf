import numpy as np 

def rdf(atoms_list, bins):

    binned = np.zeros(len(bins)-1)
    
    if not isinstance(atoms_list,list):
        atoms_list = [atoms]
    
    
    for atoms in atoms_list:
        n_atoms = int(atoms.get_number_of_atoms()/3)
        binned += np.histogram(\
            atoms.get_all_distances(mic = True)[::3,::3][np.triu_indices(n_atoms,1)],
                    bins)[0]

    return rdf

