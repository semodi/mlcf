import numpy as np 
import pandas as pd 
from ase import Atoms
from ase import Atom
from ase.io import read
from ase.io import iread
from ase.io import write
from ase.io.trajectory import TrajectoryReader
import os

os.environ['QT_QPA_PLATFORM']='offscreen'
import ipyparallel as parallel

client = parallel.Client(profile='default2')

view = client.load_balanced_view()

print(len(client.ids))

dr = 0.05
bins = np.arange(2,8,dr)


def get_binned(indices, traj_path, bins, a = 0):
    import numpy as np 
    import pandas as pd 
    from ase import Atoms
    from ase import Atom
    from ase.io import read
    from ase.io import iread
    from ase.io import write
    from ase.io.trajectory import TrajectoryReader
    import os

    def binned_distance(dist, bins):
        binned = np.zeros(len(bins)-1)
        binned += np.histogram(dist, bins)[0]
        return binned 

    dr = 0.05
    bins = np.arange(2,8,dr)

    r_oo = np.zeros(len(bins) - 1)
    
    for i in indices:
        traj = read(traj_path, index = i)
        if a > 0:
            traj.cell = [a,a,a]
            traj.pbc = True
        print(traj)
        r_oo += binned_distance(traj.get_all_distances(mic = True)[::3,::3][np.triu_indices(128,1)], bins)
    
    g_avg = .5*128*(127)/(15.646**3)
    n_obs = len(indices)
    norm = bins[:-1]**2 * g_avg * n_obs * 4 * np.pi * dr
    
    return r_oo/norm

def get_binned_oh(indices, traj_path, bins, a = 0):
    import numpy as np 
    import pandas as pd 
    from ase import Atoms
    from ase import Atom
    from ase.io import read
    from ase.io import iread
    from ase.io import write
    from ase.io.trajectory import TrajectoryReader
    import os

    def binned_distance(dist, bins):
        binned = np.zeros(len(bins)-1)
        binned += np.histogram(dist, bins)[0]
        return binned 

    dr = 0.05
    bins = np.arange(0,8,dr)

    r_oo = np.zeros(len(bins) - 1)
    
    for i in indices:
        traj = read(traj_path, index = i)
        if a > 0:
            traj.cell = [a,a,a]
            traj.pbc = True
        print(traj)
        r_oo += binned_distance(traj.get_all_distances(mic = True)[::3,1::3], bins)
        r_oo += binned_distance(traj.get_all_distances(mic = True)[::3,2::3], bins)
        
    g_avg = .5*128*(127)/(15.646**3)
    n_obs = len(indices)
    norm = bins[:-1]**2 * g_avg * n_obs * 4 * np.pi * dr
    
    return r_oo/norm

def get_binned_hh(indices, traj_path, bins, a = 0):
    import numpy as np 
    import pandas as pd 
    from ase import Atoms
    from ase import Atom
    from ase.io import read
    from ase.io import iread
    from ase.io import write
    from ase.io.trajectory import TrajectoryReader
    import os

    def binned_distance(dist, bins):
        binned = np.zeros(len(bins)-1)
        binned += np.histogram(dist, bins)[0]
        return binned 

    dr = 0.05
    bins = np.arange(0,8,dr)

    r_oo = np.zeros(len(bins) - 1)
    
    for i in indices:
        traj = read(traj_path, index = i)
        if a > 0:
            traj.cell = [a,a,a]
            traj.pbc = True
        print(traj)
        r_oo += binned_distance(traj.get_all_distances(mic = True)[1::3,2::3], bins)
    
    g_avg = .5*128*(127)/(15.646**3)
    n_obs = len(indices)
    norm = bins[:-1]**2 * g_avg * n_obs * 4 * np.pi * dr
    
    return r_oo/norm

def get_roo(basepath, start = 100, dt = 1, block_size = 1):
    
    logfile = pd.read_csv(basepath + '.log', delim_whitespace=True)
    traj_path = basepath + '.traj'
    bins = np.arange(2,8,dr)
    dt_per_block = int(block_size/dt)
    end = int((len(logfile) - start)/block_size) * block_size + start

    indices = np.array(range(start,end,dt))
    indices = indices.reshape(-1, dt_per_block)

    r_oo = view.map_sync(get_binned, indices, [traj_path]*len(indices), [bins]*len(indices))
    return np.array(r_oo)

def get_roh(basepath, start = 100, dt = 1, block_size = 1):
    
    logfile = pd.read_csv(basepath + '.log', delim_whitespace=True)
    traj_path = basepath + '.traj'
    bins = np.arange(2,8,dr)
    dt_per_block = int(block_size/dt)
    end = int((len(logfile) - start)/block_size) * block_size + start

    indices = np.array(range(start,end,dt))
    indices = indices.reshape(-1, dt_per_block)

    r_oo = view.map_sync(get_binned_oh, indices, [traj_path]*len(indices), [bins]*len(indices))
    return np.array(r_oo)

def get_rhh(basepath, start = 100, dt = 1, block_size = 1):
    
    logfile = pd.read_csv(basepath + '.log', delim_whitespace=True)
    traj_path = basepath + '.traj'
    bins = np.arange(2,8,dr)
    dt_per_block = int(block_size/dt)
    end = int((len(logfile) - start)/block_size) * block_size + start

    indices = np.array(range(start,end,dt))
    indices = indices.reshape(-1, dt_per_block)

    r_oo = view.map_sync(get_binned_hh, indices, [traj_path]*len(indices), [bins]*len(indices))
    return np.array(r_oo)

def get_roo_xyz(basepath, a, start = 100, dt = 1, block_size = 1):
    
    logfile = read(basepath + '.xyz', index = ':')
    traj_path = basepath + '.xyz'
    bins = np.arange(2,8,dr)
    dt_per_block = int(block_size/dt)
    end = int((len(logfile) - start)/block_size) * block_size + start

    indices = np.array(range(start,end,dt))
    indices = indices.reshape(-1, dt_per_block)

    r_oo = view.map_sync(get_binned, indices, [traj_path]*len(indices), [bins]*len(indices),[a]*len(indices))
    return np.array(r_oo)


def blocking(obs):
    n0 = len(obs)
    max_power = int(np.log(n0/2)/np.log(2))
    blocks = 2**np.arange(max_power)
    
    c0 = []
    
    for b in blocks:
        n_blocks = int(len(obs)/b)
        trunc_len = int(len(obs)/b) * b
        
        c0.append(np.std(np.mean(obs[:trunc_len].reshape(n_blocks,-1,obs.shape[1]),axis=1),axis = 0))
     
    c0 = np.array(c0)
    n = (n0/np.array(blocks)).astype(int)
    n = np.tile(n.reshape(-1,1),[1,c0.shape[1]])
    return blocks, np.sqrt(c0/(n - 1)), np.sqrt(c0/(n - 1))*(1/np.sqrt(2*(n-1)))
    
    

def t_corr(obs, t_range):
    avg = np.mean(obs, axis = 0)
    n0 = len(obs)
    corr = []
    
    for t in t_range:
        obs2 =  obs[t:]
        obs1 = obs[:len(obs2)]
        corr.append(np.mean((obs2-avg)*(obs1-avg),axis = 0))
    
    return t_range, corr
    