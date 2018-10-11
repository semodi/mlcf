import mlc_func.elf as elf
import sys
import json
import os

os.environ['QT_QPA_PLATFORM']='offscreen'

if __name__ == '__main__':

    root = sys.argv[1]
    basis = json.loads(open('./basis.json','r').readline())
    method = basis['alignment']
    if len(sys.argv) > 2:
        profile = sys.argv[2]

        elf.utils.preprocess_all(root, basis, method=method,
             view = elf.utils.get_view(profile))
    else:
        elf.utils.preprocess_all(root, basis, method=method)
