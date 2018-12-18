default_settings = {'solutionmethod': ['diagon','omm'],
                'basis': ['dz_custom','dz','qz_custom','sz','uf','szp'],
                'xcfunctional': ['bh','pbe','pw92','revpbe'],
                'model': 'None',
                'mixing' : False,
                'cmcorrection' : False,
                'ipp_client': 'None'
                }

default_mixing = {'model1': 'None',
                  'model2': 'None',
                  'basis1': ['uf','dz_custom','qz_custom','dz','sz','szp'],
                  'basis2': ['dz_custom','dz','qz_custom','sz','uf','szp'],
                  'xcfunctional1': ['pbe','bh','pw92','revpbe'],
                  'xcfunctional2': ['bh','pbe','pw92','revpbe'],
                  'solutionmethod1': ['diagon','omm'],
                  'solutionmethod2': ['diagon','omm'],
                  'n': 5,
                  'cmcorrection1' : False,
                  'cmcorrection2' : False,
                  'which' : ''
                 }
