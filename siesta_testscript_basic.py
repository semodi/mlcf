import shutil
import os
import shutil
import subprocess
import sys
#TODO: Get rid of absolute paths
os.environ['QT_QPA_PLATFORM']='offscreen'

if __name__ == '__main__':
    n_clients = int(sys.argv[1])
    try:
        os.chdir('/gpfs/scratch/smdick/siesta/')
    except FileNotFoundError:
        os.mkdir('/gpfs/scratch/smdick/siesta/')
        os.chdir('/gpfs/scratch/smdick/siesta/')

    try:
        for i in range(n_clients):
            shutil.os.mkdir('{}'.format(i))
    except FileExistsError:
        pass

    for id in range(n_clients):
        if id == n_clients - 1:
            subprocess.call('python /gpfs/home/smdick/md_routines/prop_and_eval.py {}'.format(id), shell = True)
        else:
            subprocess.call('python /gpfs/home/smdick/md_routines/prop_and_eval.py {} &'.format(id), shell = True)
        
    print('done')

    
