""" Script to transform electron density of all systems contained in a directory
into electonic descriptors that can be used for machine learning

to run:

python preprocess_all.py root_dir [ipp_profile [n_atoms]]

root_dir: root directory containing all systems (systems/files should have unique labels)
ipp_profile: profile for ipyparallel client
n_atoms: only calculate descriptors for the first n_atoms of each system (default
    is to calculate all descriptors)
"""
import mlc_func.elf as elf
import sys
import json
import os

os.environ['QT_QPA_PLATFORM']='offscreen'

if __name__ == '__main__':

    root = sys.argv[1]
    basis = json.loads(open('./basis.json','r').readline())
    method = basis['alignment']

    n_atoms = -1
    if len(sys.argv) > 3:
        n_atoms = int(sys.argv[3])
        print(n_atoms)

    if len(sys.argv) > 2:
        profile = sys.argv[2]
        print(profile)

        elf.utils.preprocess_all(root, basis, method=method,
             view = elf.utils.get_view(profile), n_atoms = n_atoms)
    else:
        elf.utils.preprocess_all(root, basis, method=method, n_atoms = n_atoms)
