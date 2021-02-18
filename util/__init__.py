import json
import torch
from torch.utils import data

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def get_config(filepath):
    config = {}
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config

def create_filename_suffix(data_type, batch_size, instrument_list, instrument_count, learning_rate, instrument_source):
    suffix = ''

    instr_abbr = ''
    instr_abbr = ''.join([i[0] for instr in instrument_list for i in instr.split()])
    
    # suffix = '-' + instr_abbr + '-' + str(instrument_count)
    if learning_rate is None:
        suffix = '-{}-{}'.format(instr_abbr, instrument_count)
    if learning_rate is not None:
        print('{:f}'.format(learning_rate))
        print('{:f}'.format(learning_rate).split('.')[1])
        suffix = '-{}-{}-{}-{}-{}-lr{}'.format(instrument_source, data_type, batch_size, instr_abbr, instrument_count, '{:f}'.format(learning_rate).split('.')[1])
    return suffix