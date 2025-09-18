# Standard libraries
import csv
import os

# 3rd party
import h5py
from mendeleev import element
import numpy as np

def read_data(interaction, matter_types=[]):
    '''
    Reads nucleus and matter data into a list, where each element is
    a dictionary containing the information about the datum.
    '''
    data = []

    # Read nuclei data and put in a useful format
    fname = get_fname_nuclei(interaction)
    print(f'Reading MBPT data from {fname} ...')
    f = open(fname)
    csv_file = csv.DictReader(f)
    for line in csv_file:
        line = clean_line(line)
        d = replace_keys(line)
        d = casting(d)
        d = add_info(d, interaction)
        d['matter_type'] = None
        data.append(d)
    f.close()
    
    # Sort by mass number
    data.sort(key=lambda x: x.get('A'))

    # Add nuclear matter to the data if that's requested
    densities = [0.08, 0.16]
    for matter_type in matter_types:
        for density in densities:
            d = matter_entry(interaction, matter_type, density)
            data.append(d)
            print(f"Added {d['label']} to read data")
    print('Done reading data.')
    return data

def matter_entry(interaction, matter_type, density):
    '''Create a data point for nuclear matter.'''
    fname = get_fname_matter(interaction, matter_type)
    f = h5py.File(fname, 'r')
    n0 = 0.16
    d = {}
    d['n'] = density
    d['matter_type'] = matter_type
    density_n0 = density / n0
    d['label'] = f'{matter_type}_{density_n0:.1f}n0'
    tmp = rf'{density_n0:.1f}n_0'
    d['pretty_label'] = rf'{matter_type}$_{{{tmp}}}$'
    densities = f.get('n')[...]
    idx = None
    for i, tmp in enumerate(densities):
        # Find the index of the requested density
        if np.isclose(tmp, density):
            idx = i
    if idx is None:
        raise ValueError(f'Density {density} not found in {fname}')
    d['EHF'] = f.get('HF')[idx]
    d['MBPT(2)'] = f.get('mbpt2')[idx]
    d['MBPT(3)'] = f.get('mbpt3')[idx]
    d['interaction'] = interaction
    d['pretty_interaction'] = pretty_interaction(interaction)
    f.close()
    return d

def add_info(d, interaction):
    '''Add some useful information to the datum d.'''
    d['interaction'] = interaction
    d['pretty_interaction'] = pretty_interaction(interaction)
    Z = d.get('Z')
    A = d.get('A')
    d['label'] = element(Z).symbol + f'{A}'
    d['pretty_label'] = pretty_isotope(Z, A)
    return d

def get_data_path():
    '''The path to where the data is stored.'''
    return 'data'

def get_fname_nuclei(interaction):
    '''Get filename for the finite nuclei data.'''
    path = get_data_path()
    if interaction == 'EM1.8_2.0':
        return f'{path}/1820_EM.csv'
    elif interaction == 'DNNLOgo':
        return f'{path}/DeltaNNLOgo_394.csv'
    elif interaction == 'EM7.5':
        return f'{path}/1820_EM75.csv'
    else:
        raise ValueError

def get_fname_matter(interaction, matter_type):
    '''Get filename for the nuclear matter data.'''
    assert(matter_type in ['PNM', 'SNM'])
    if interaction == 'EM1.8_2.0':
        return f'{get_data_path()}/1820_EM_{matter_type}.h5'
    else:
        raise ValueError

def replace_keys(d):
    '''Change some data labels.'''
    d['EHF'] = d.pop('E_HF')
    d['MBPT(2)'] = d.pop('E_MP2')
    d['MBPT(3)'] = d.pop('Ecorr_MP3')
    d['IMSRG'] = d.pop('E_HF + E_IMSRG')
    return d

def pretty_isotope(Z, A):
    '''Get a nice label for a nucleus, useful for plotting.'''
    s = element(Z).symbol
    return rf'$^{{{A}}}${s}'

def pretty_interaction(interaction):
    '''Get a nice variant of the interaction, useful for plotting.'''
    if interaction == 'DNNLOgo':
        return r'$\Delta$N$^2$LO$_{\textnormal{GO}}$'
    elif interaction == 'EM1.8_2.0':
        return '1.8/2.0 (EM)'
    elif interaction == 'EM7.5':
        return '1.8/2.0 (EM7.5)'
    else:
        raise ValueError(f'Unknown interaction {interaction}')

def casting(data):
    '''Change data type for some read data.'''
    ints = ['Z', 'A', 'emax', 'e3max', 'hw']
    floats = ['EHF', 'MBPT(2)', 'MBPT(3)', 'IMSRG']
    for key in floats:
        data[key] = safecast(float, data.get(key))
    for key in ints:
        data[key] = safecast(int, data.get(key))
    return data

def safecast(cast_to, value):
    try:
        return cast_to(value)
    except (TypeError, ValueError):
        return None

def clean_line(line):
    return {k.strip():v.strip() for k, v in line.items()}

def filter_nuclei(data, nuclei):
    '''Removes data points from the data, only keeping those specified in nuclei.'''
    if 'all' in nuclei:
        return data
    new_data = []
    for d in data:
        if d.get('label') in nuclei:
            new_data.append(d)
    return new_data

def get_nucleus_dict(data, Z, A):
    '''Get the dictionary for a certain nucleus.'''
    for i in range(len(data)):
        d = data[i]
        if d.get('Z') == Z and d.get('A') == A:
            return d

def print_nuclei(data):
    for d in data:
        print(d.get('label'), end=' ')
    print()

def get_order(order, Eformat=False, enforce_int=False, enforce_str=False):
    '''Convert, e.g., "MBPT(2)" to "1" and vice versa.'''
    val = None
    assert(not (enforce_int and (Eformat or enforce_str)))
    is_strformat = type(order) == str
    is_intformat = type(order) == int
    if is_strformat:
        # Convert string to int or E-format, or ensure "MBPT(x)" is returned
        order = order.upper()
        if order == 'EHF':
            val = 'EHF' if enforce_str or Eformat else 0
        elif order == 'MBPT(2)' or order == 'E2':
            val = 'MBPT(2)' if enforce_str else 1
            val = 'E2' if Eformat else val
        elif order == 'MBPT(3)' or order == 'E3':
            val = 'MBPT(3)' if enforce_str else 2
            val = 'E3' if Eformat else val
        elif order == 'MBPT(4)' or order == 'E4':
            val = 'MBPT(4)' if enforce_str else 3
            val = 'E4' if Eformat else val
        else:
            raise ValueError(f'Unknown order {order}')
    elif is_intformat:
        # Convert int to string
        if order == 0:
            val = 'EHF'
        elif order == 1:
            val = 'E2' if Eformat else 'MBPT(2)'
        elif order == 2:
            val = 'E3' if Eformat else 'MBPT(3)'
        elif order == 3:
            val = 'E4' if Eformat else 'MBPT(4)'
        else:
            raise ValueError(f'Unknown order {order}')
        if enforce_int:
            # Nothing to do, "order" is already an int
            val = order
    else:
        raise ValueError(f'Cannot convert type {type(order)}')
    # Extra safety checks
    assert(type(val) == int if enforce_int else True)
    assert(type(val) == str if enforce_str else True)
    assert(val[0] == 'E' if Eformat else True)
    if (type(order) == int and order == 0) or (type(order) == str and 'HF' in order):
        assert(val == 'EHF' if enforce_str else True)
    else:
        assert(val[0] == 'M' if not Eformat and enforce_str else True)
    return val

def get_order_list(minorder, maxorder):
    low = get_order(minorder, enforce_int=True)
    high = get_order(maxorder, enforce_int=True)
    return [k for k in range(low, high+1)]

def get_array(data, key, aslist=False):
    '''Get the requested information for all data points.'''
    n = len(data)
    arr = []
    for i in range(n):
        arr.append(data[i].get(key))
    if not aslist:
        arr = np.array(arr)
    return arr

def get_common_value(data, key):
    '''Read a value that is common to all data points, e.g. the interaction.'''
    val = data[0].get(key)
    for d in data:
        assert(val == d.get(key))
    return val

def get_energy(data, order):
    '''Read MBPT energy from the data.'''
    order = get_order(order, enforce_str=True)
    return get_array(data, order)
