import elf
import sys
import json

if __name__ == '__main__':

    root = sys.argv[1]
    basis = json.loads(open('./basis.json','r').readline())
    method = basis['alignment']
    if len(sys.argv) > 3:
        profile = sys.argv[3]

        elf.utils.preprocess_all(root, basis, method=method,
             view = elf.utils.get_view(profile))
    else:
        elf.utils.preprocess_all(root, basis, method=method)
