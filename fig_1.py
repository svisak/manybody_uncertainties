# Standard libraries
import argparse

# Local
import inout
import plotting

parser = argparse.ArgumentParser()
parser.add_argument('-dir', '--output_dir', type=str)
args = parser.parse_args()
output_dir = args.output_dir if args.output_dir is not None else 'output'

interaction = 'EM1.8_2.0'
orders = ['MBPT(2)', 'MBPT(3)']
data = inout.read_data(interaction, matter_types=['SNM'])
plotting.R_data_paper(interaction, data, output_dir, orders=orders)
