default_settings = {'xyzpath': '/',
                'name': 'md_siesta',
                'integrator': ['vv','nh','none'],
                't': 300,
                'dt': 0.5, #fs
                'Nt' : 1000000,
                'ttime': 10,
                'restart': False,
                'solutionmethod': ['diagon','omm'],
                'basis': ['dz_custom','dz','sz','uf'],
                'xcfunctional': ['bh','pbe','pw92','revpbe'],
                'model': '/',
                'modelkind': ['none','atomic','molecular','mulliken','mbpol'],
                'mixing' : False,
                'cmcorrection' : False
                }

default_mixing = {'model1': '/',
                  'model2': '/',
                  'basis1': ['uf','dz_custom','dz','sz'],
                  'basis2': ['dz_custom','dz','sz','uf'],
                  'xcfunctional1': ['pbe','bh','pw92','revpbe'],
                  'xcfunctional2': ['bh','pbe','pw92','revpbe'],
                  'modelkind1': ['none','atomic','molecular','mulliken','mbpol'],
                  'modelkind2': ['none','atomic','molecular','mulliken','mbpol'],
                  'solutionmethod1': ['diagon','omm'],
                  'solutionmethod2': ['diagon','omm'],
                  'n': 5
                 }
