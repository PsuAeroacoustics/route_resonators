#!/usr/bin/env python3
import os
from funcs import *

#%%

def main():
    
	case_dir = os.getcwd()
	saved_params = read_results_from_h5(case_dir)
	
	if not 'blade_nodes' in saved_params:
		build_blade_geom(saved_params)

	initialize_grid(saved_params)
	arange_resonators(saved_params,uniform=True,preroute = True)
	res_paths = route_resonators(saved_params,randomize=True)
	wrtie_res_paths(res_paths)
	print('All packed!')


if __name__ == "__main__":
	main()
	print("exiting main.py")