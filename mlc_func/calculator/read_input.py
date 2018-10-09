import sys
from .default_settings import *

def read_input(input_file):
    """ Read provided input file for settings. Where none are given,
    use values specified in default_settings
    """
    settings = {}
    mixing_settings = {}

    # Set settings equal to default settings.
    # If multiple options are listed in default settings than the first item is
    # used as default
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

                # Hack to cast a string to boolean (otherwise any non-empty string
                # is interpreted as True)
                if isinstance(s[key], bool):
                    s[key] = {'false': False, 'true': True}[value]
                else:
                    # Default value defines the type of item
                    s[key] = type(s[key])(value)
            else:
                raise Exception('Invalid key {} in input file'.format(key))
            line = file.readline()

    for s in settings:
        if settings[s] in ['none','None']: settings[s] = None
    for s in mixing_settings:
        if mixing_settings[s] in ['none','None']: mixing_settings[s] = None

    for key in settings:
        print( '({}, {})'.format(key, settings[key]))
    if settings['mixing']:
        for key in mixing_settings:
            print( '({}, {})'.format(key, mixing_settings[key]))

    return settings, mixing_settings
