import sys
from default_settings import *

if len(sys.argv) < 2:
    raise Exception('Please provide input file')

input_file = sys.argv[1]

settings = {}
mixing_settings = {}

for s,d in zip([settings, mixing_settings], [default_settings, default_mixing]):
    for key in d:
        if isinstance(d[key],list):
            s[key] = d[key][0]
        else:
            s[key] = d[key]

mixing_block = False

with open(input_file, 'r') as file:
    line = file.readline()
    while(line):

        if line == '':
            line = file.readline()
            continue

        if '%block mixing' in line:
            mixing_block = True
            settings['mixing'] = True
            line = file.readline()
            continue
        elif '%endblock mixing' in line:
            mixing_block = False
            line = file.readline()
            continue

        try:
            key, value = line.split()
        except ValueError:
            line = file.readline()
            continue

        key = key.lower()
        if not 'model' in key or 'xyzpath' in key:
            value = value.lower()

        s = [settings, mixing_settings][int(mixing_block)]
        d = [default_settings, default_mixing][int(mixing_block)]

        if key in s:
            if isinstance(d[key],list):
                if not value in d[key]:
                    raise Exception( '({}, {}) not a valid option'.format(key, value))
            s[key] = type(s[key])(value)
        else:
            raise Exception('Invalid key {} in input file'.format(key))
        line = file.readline()

print(settings)
print(mixing_settings)
