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
