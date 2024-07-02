#!/usr/bin/env python3
import os
from funcs import *

#%%

def main():
    
	case_dir = os.getcwd()
	saved_params = read_results_from_h5(case_dir)
	
	if not 'blade_nodes' in saved_params:
		build_blade_geom(saved_params)

	generate_domain(saved_params)
	arange_resonators(saved_params)
	res_paths = route_resonators(saved_params)
	print('All packed!')


if __name__ == "__main__":
	main()
	print("exiting main.py")