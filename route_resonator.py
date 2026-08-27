#!/usr/bin/env python3
import argparse
from funcs import *

#%%

def main():
	
	parser = argparse.ArgumentParser("Generic resonator routing script", description="This script routes open-closed cavities inside of an arbitrary geometry.")
	parser.add_argument("-input_params",type= str,required=False,default="input_params.json5",help = "Name of the json5 input file")
	parser.add_argument("-stl_file",type= str,required=False,help="Name of the stl file including the extension")
	args = parser.parse_args()
	

	input_params = read_input_params(args)
	saved_params = {}
	import_geom(input_params,saved_params)	
	initialize_domain(input_params,saved_params)

	arange_resonators(input_params,saved_params,uniform=True)
	res_paths = route_resonators(input_params,saved_params)
	write_res_paths(res_paths)
	print('All packed!')

if __name__ == "__main__":
	main()
	print("exiting main.py")
# %%
