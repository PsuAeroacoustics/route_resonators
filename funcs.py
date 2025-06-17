#!/usr/bin/env python3
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# import aerosandbox as asb
import h5py
import os
from random import randint
from shutil import rmtree
import trimesh
import pyjson5
from scipy.spatial import cKDTree
import pyvista as pv

#%%

surrounding_node_direction = np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])

def get_index(pnts,saved_params):
    return tuple(np.round((pnts-saved_params['grid_bounds'][::2])/saved_params['dx']).astype(int).T)

def read_results_from_h5(case_dir):
    saved_params ={}
    with h5py.File(os.path.join(case_dir, 'saved_params.h5'), 'r') as f:
        for k,v in f.items():
            if isinstance(v[()], bytes):
                saved_params.update({k:v[()].decode()})
            else:
                saved_params.update({k:v[()]})
    return saved_params


def read_input_params(args):

    with open(args.input_params) as input_file:
        input_params = pyjson5.load(input_file)
    
    if args.stl_file is None:
        assert "STL_file" in input_params, "Name of the STL file needs to be provided either as an command line argument or in the input file"
    else:
        input_params['STL_file'] = args.stl_file
        
    return input_params

def import_geom(input_params,saved_params):

    # geom = pv.read(input_params['STL_file'])
    # geom = geom.clean()
    geom = pv.Box(bounds=input_params['dimensions'], level=4, quads=True)
    # Offset the face centers
    offset_centers = geom.points - input_params['shell_thickness'] * geom.point_normals
    # Create a new mesh with the displaced face centers
    geom_offset = pv.PolyData(offset_centers,faces = geom.faces)

    
    saved_params.update({'geom':geom,'geom_offset':geom_offset})

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(grid_coord[grid][::10,0],grid_coord[grid][::10,1],grid_coord[grid][::10,2])
    # ax.scatter(grid_coord[np.invert(grid)][::10,0],grid_coord[np.invert(grid)][::10,1],grid_coord[np.invert(grid)][::10,2])

    # # ax.scatter(offset_mesh.points[:,0],offset_mesh.points[:,1],offset_mesh.points[:,-1])
    # # ax.scatter(offset_mesh.points[:,0],offset_mesh.points[:,1],offset_mesh.points[:,-1])

    # # ax.scatter(vertices_offset[:,0],vertices_offset[:,1],vertices_offset[:,-1])

    # # ax.scatter(mesh.vertices[res_ind,0],mesh.vertices[res_ind,1],mesh.vertices[res_ind,-1])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.grid(False)
    # ax.xaxis.pane.fill = False 
    # ax.yaxis.pane.fill = False 
    # plt.show(block=True)


def initialize_domain(input_params,saved_params):

    dx = 1.1*(2*input_params['a'])

    x_min,y_min,z_min = saved_params['geom'].bounds[::2]
    x_max,y_max,z_max = saved_params['geom'].bounds[1::2]

    Nx = np.ceil((x_max-x_min+2*dx)/dx)
    x = np.arange(Nx)*dx+x_min-dx
    Ny = np.ceil((y_max-y_min+2*dx)/dx)
    y = np.arange(Ny)*dx+y_min-dx
    Nz = np.ceil((z_max-z_min+2*dx)/dx)
    z = np.arange(Nz)*dx+z_min-dx
    
    grid_coord = pv.RectilinearGrid(x, y, z)
    grid_bounds = grid_coord.bounds
    nx, ny, nz = grid_coord.dimensions

    # Compute the SDF
    sdf,sdf_offset = list(map(lambda x: grid_coord.compute_implicit_distance(x),[saved_params['geom'],saved_params['geom_offset']]))
    grid =  (sdf['implicit_distance'] < 0) & (sdf_offset['implicit_distance'] < 0)
    geom_interp = sdf.contour([0])
    initial_unoccupied_nodes = np.sum(grid)

    grid_coord = grid_coord.points.reshape((nx, ny, nz,3),order = 'F')
    grid = grid.reshape((nx, ny, nz),order = 'F')

    saved_params.update({'x':x,'y':y,'z':z,'dx':dx,'grid_coord':grid_coord,'grid':grid,'grid_bounds':grid_bounds,'geom_interp':geom_interp,'initial_unoccupied_nodes':initial_unoccupied_nodes})


def arange_resonators(input_params,saved_params, uniform = True):
    
    res_surface = saved_params['geom_interp'].points[saved_params['geom_interp'].points[:,-1]==saved_params['geom'].points[:,-1].max()] 

    row_ind = np.where(np.diff(res_surface[:,1])!=0)[0]+1
    res_surface = res_surface.reshape((int(len(res_surface)/(len(row_ind)+1)),len(row_ind)+1,3),order = 'F')

    border = np.max((input_params["border"]/input_params["shell_thickness"],1))*input_params["shell_thickness"]
    border_ind = int(np.ceil((border-saved_params['dx']/2)/saved_params['dx']+1))

    l = np.abs(res_surface[-1,0]-res_surface[0,0]).max()
    w = np.abs(res_surface[0,-1]-res_surface[0,0]).max()
    
    N_total = input_params['OAR']*l*w/(np.pi*input_params['a']**2)
    N_max = ((l-2*border_ind*saved_params['dx'])/saved_params['dx']-1)*((w-2*border_ind*saved_params['dx'])/saved_params['dx']-1)
    N = np.min((N_total,N_max))
    OAR_max = np.floor(N_max)*np.pi*input_params['a']**2/(l*w)
    print(f'Maximum OAR: {np.round(OAR_max,3)}')
    
    res_surface_trimmed = res_surface[border_ind:-border_ind,border_ind:-border_ind]
    Nx,Ny = res_surface_trimmed.shape[:2]

    if input_params['uniform']:

        skip_ind = int(np.round(np.sqrt(Nx*Ny/N)))  
        Nx,Ny = res_surface_trimmed[::skip_ind,::skip_ind].shape[:2]

        border_ind = border_ind+int(np.round((2*(Nx+Ny)-np.sqrt(4*(Nx+Ny)**2-16*(Nx*Ny-N)))/8))
        res_nodes_temp = res_surface[border_ind:-border_ind,border_ind:-border_ind][::skip_ind,::skip_ind]
        Nx,Ny = res_nodes_temp.shape[:2]
        N = int(Nx*Ny)
        OAR = N*np.pi*input_params['a']**2/(l*w)
        print(f'Actual OAR: {np.round(OAR,3)}')
        res_nodes_temp = res_nodes_temp.reshape((N,3),order = 'C')

    else:
        active_pnts = []
        res_nodes_temp = []
        grid = np.zeros(res_surface.shape[:2],dtype = bool)
        grid[border_ind:-border_ind,border_ind:-border_ind] = True
        r = np.sqrt(2)*saved_params['dx']
        x0_ind = tuple((np.array(res_surface.shape[:2])/2).astype(int))
        grid[x0_ind]= False
        active_pnt = res_surface[x0_ind]
        active_pnts.append(res_surface[x0_ind])

        while len(active_pnts)>=1 and len(res_nodes_temp)<N:
            active_pnt = active_pnts.pop(randint(0,len(active_pnts)-1))
            # computes distance from the active point to the surrounding points
            active_pnt_dist = np.linalg.norm(res_surface[grid]-active_pnt,axis = -1)
            test_pnt_ind = (active_pnt_dist<=2*r) & (active_pnt_dist >= r)
            test_pnts = res_surface[grid][test_pnt_ind]
            
            for test_pnt in test_pnts:
                test_pnt_dist = np.linalg.norm(res_surface[grid]-test_pnt,axis = -1)
                if np.all(grid[grid][test_pnt_dist<=r]):
                    grid[get_index(res_surface[grid][test_pnt_dist==0],saved_params)[:2]] = False
                    active_pnts.append(test_pnt)
                    res_nodes_temp.append(test_pnt)  
        
        res_nodes_temp = np.array(res_nodes_temp)[:int(N)]
        for i in range(2):
            sort_ind = res_nodes_temp[:,1-i].argsort(kind = 'stable')
            res_nodes_temp = res_nodes_temp[sort_ind]
        ind = np.where(np.diff(res_nodes_temp[:,0])>0)[0]+1
        ind = np.insert(ind,(0,len(ind)),(0,len(res_nodes_temp)))
        res_nodes_temp = [res_nodes_temp[ind[i]:ind[i+1]] for i in range(len(ind)-1)]
        
    # if input_params['pre_route']:
    #     res_nodes = []
    #     # nodes = np.sort(res_nodes_temp[res_nodes_temp[:,1]==res_nodes_y[y_iter]],axis=0, kind='stable')
    #     if input_params['uniform']:
    #         for x_iter in range(Nx):
    #             initial_depth = np.max((np.sum(saved_params['grid'][get_index(res_nodes_temp[x_iter][0],saved_params)[:2]]),len(res_nodes_temp[x_iter])))
    #             for node_iter,node in enumerate(res_nodes_temp[x_iter]):
    #                 node_ind = get_index(node,saved_params)
    #                 N_nodes = initial_depth-node_iter
    #                 starting_nodes = saved_params['grid_coord'][node_ind[:2]][saved_params['grid'][node_ind[:2]]][::-1][:N_nodes]
    #                 saved_params['grid'][get_index(starting_nodes,saved_params)] = False
    #                 res_nodes.append(np.insert(starting_nodes,0, node,axis = 0))
    # else:
    res_nodes = []
    for i,node in enumerate(res_nodes_temp):
        starting_nodes_ind = get_index(node,saved_params)
        starting_nodes = saved_params['grid_coord'][starting_nodes_ind[:2]][saved_params['grid'][starting_nodes_ind[:2]]]
        if input_params['pre_route']:
            init_length =np.min((len(starting_nodes)-1,Nx))
            starting_nodes = starting_nodes[-(init_length-i%Nx+1):][::-1]
        else:
            starting_nodes = starting_nodes[-1]
        saved_params['grid'][get_index(starting_nodes,saved_params)] = False
        res_nodes.append(np.vstack((node,starting_nodes)))

    N_res = len(input_params['L'])
    if N_res>1:
        if  input_params['uniform']:
            res_type = np.zeros(N_res,dtype=int)
            res_type[::2] = np.arange(np.ceil(N_res/2))
            res_type[1::2] = np.arange(np.floor(N_res/2))+np.ceil(N_res/2)
            res_type = np.tile(res_type,int(np.ceil(Ny/N_res)))[:Ny]
            res_type = np.array([np.roll(res_type,-(ii%2)*int(N_res/2)) for ii in range(int(Nx))]).ravel()
        else:
            res_type = np.random.permutation(np.tile(np.arange(N_res),int(np.ceil(N/N_res)))[:N])
    else:
        res_type = np.zeros(N,dtype = int)
    
    if N_res>1:
        route_order_ind = np.array(saved_params['res_types']).argsort(kind = 'stable')[::-1]
    else:
        # route_order_ind  = np.arange(N)
        route_order_ind = np.random.permutation(np.arange(N))
        # route_order_ind  = np.arange(N).reshape(Nx,Ny,order = 'F').ravel()

    saved_params.update({'res_nodes':res_nodes,'res_types':res_type,'route_order_ind':route_order_ind})


def route(starting_node,total_length,saved_params):

    # starting length determined by the distance between the starting point and the nearest point inside the geometry 
    L = np.abs(starting_node[-1]-starting_node[0])[-1]
    # starting direction downwards in z
    active_direction = np.array([0,0,-1])
    # initializes open and closed sets with the position of the starting nodes
    closed_set = []
    closed_set.extend(starting_node)

    packed = False
    while len(closed_set) > 1 and L>0:
        print(total_length-L)

        # Checks if total length requirement is satisfied 
        if (total_length-L)<=0:
            packed = True
            trim_ind = int(np.round((total_length-(L-saved_params['dx']*(len(closed_set)-1)))/saved_params['dx']))
            saved_params['grid'][get_index(np.array(closed_set[trim_ind:]),saved_params)] = True
            closed_set = np.asarray(closed_set[:trim_ind])
            print('Path found!')
            break

        else:
            
            # removes last element in the open set 
            active_node = closed_set[-1]
            # computes the corresponding index
            active_node_ind = get_index(active_node,saved_params)

            # list of all points in either direction in x,y,z
            surrounding_nodes = [saved_params['grid_coord'][:active_node_ind[0],active_node_ind[1],active_node_ind[2]][::-1],
                         saved_params['grid_coord'][active_node_ind[0]+1:,active_node_ind[1],active_node_ind[2]],
                         saved_params['grid_coord'][active_node_ind[0],:active_node_ind[1],active_node_ind[2]][::-1],
                         saved_params['grid_coord'][active_node_ind[0],active_node_ind[1]+1:,active_node_ind[2]],
                         saved_params['grid_coord'][active_node_ind[0],active_node_ind[1],:active_node_ind[2]][::-1],
                         saved_params['grid_coord'][active_node_ind[0],active_node_ind[1],active_node_ind[2]+1:]]

            # removes nodes in the direction opposite to the current growth direction
            surrounding_nodes.pop(int(np.where(np.all(surrounding_node_direction==-active_direction,axis = -1))[0]))
            # initializes list for storing the candidate points
            candidate_nodes = []
            # iterates through each direction defined by the surrounding points
            for pnts in surrounding_nodes:
                # will only append the points in a certain direction if the two points adjacent to the current point are unoccupied
                if len(pnts)>0 and np.all(saved_params['grid'][get_index(pnts[:1],saved_params)]):
                    # # will only append the points that
                    # if (get_index(pnts[0]) - active_node_ind) != -current_direction:
                    ind = get_index(pnts,saved_params)
                    # retains all unoccupied nodes 
                    max_ind = np.where(np.invert(saved_params['grid'][ind]))[0][0]
                    candidate_nodes.append(saved_params['grid_coord'][ind][:max_ind])

            # if avaliable surrounding points exist
            if len(candidate_nodes)>0:                
                # selects the candidate points based on distance and direction - priority will be given to the current direction of growth followed by the greatest distance to the next occupied node
                if len(candidate_nodes)==1:
                    candidate_nodes = candidate_nodes[0]
                else:
                    # gets candidate directions 
                    candidate_directions = np.array([get_index(nodes[0],saved_params)-np.array(active_node_ind) for nodes in candidate_nodes])
                    # determines index of direction that corresponds to current direction 
                    active_direction_ind = np.where(np.all(candidate_directions == active_direction,axis = -1))[0]
                
                    if len(active_direction_ind)!=0:
                        candidate_nodes = candidate_nodes[int(active_direction_ind)]
                    else:
                        # selects the directions with the greatest distance to the next occupied node 
                        dist_sort_ind = np.array([len(nodes) for nodes in candidate_nodes]).argmax()
                        candidate_nodes = candidate_nodes[int(dist_sort_ind)]
                                
                # # appends all the candidate nodes to the open set in order of packing priority
                # open_set.extend(candidate_nodes)
                # appends only the candidate nodes with the highest priority to the closed set
                closed_set.extend(candidate_nodes)

                # updates background grid with newly occupied nodes
                saved_params['grid'][get_index(candidate_nodes,saved_params)] = False
                # updates current direction of growth
                active_direction = get_index(candidate_nodes[0],saved_params)-np.array(active_node_ind)
                # updates total routed length
                L = L+len(candidate_nodes)*saved_params['dx']

            # if all adjescent points are occupied and a candidate directions doesn't exist aka dead end
            else:
                if len(closed_set)>3:    

                    directions = np.diff(get_index(np.array(closed_set[-3:][::-1]),saved_params),axis = -1).T
                    # updates the current direction as being reversed
                    active_direction = directions[0]

                    rm_iter =   [1 if np.all(directions[0] ==directions[-1]) else 2][0]
                    for i in range(rm_iter):
                        # removes the last element from the closed set
                        rm_node_ind = closed_set.pop()
                        # resets that point as being unoccupied on the background grid
                        saved_params['grid'][get_index(rm_node_ind,saved_params)] = True
                        # updates total routed length
                        L = L-saved_params['dx']*rm_iter
                else:
                    break
    if not packed:
        saved_params['grid'][get_index(np.array(closed_set)[1:],saved_params)] = True
        closed_set = []
    return closed_set

def route_resonators(input_params,saved_params):
   
    res_paths = []

    for i in range(len(saved_params['res_nodes'])):
        res_paths.append(route(starting_node = saved_params['res_nodes'][saved_params['route_order_ind'][i]],total_length = input_params['L'][saved_params['res_types'][saved_params['route_order_ind'][i]]],saved_params = saved_params))

    r_bend = np.min((input_params['r_bend'],saved_params['dx']))
    
    res_paths = list(filter(len,res_paths))
    N = np.zeros(len(res_paths))
    L = np.zeros(len(res_paths))

    for i,res_path in enumerate(res_paths):
        N[i] = len(res_path)
        L[i] = np.diff(res_path[:2],axis = 0)[0][-1]+(len(res_path)-1)*saved_params['dx']
        bend_ind = np.where(np.any(np.abs(np.diff(np.diff(res_path,axis = 0),axis = 0))>=0.9*saved_params['dx'],axis = -1))[0]+1
        center_pnt = res_path[bend_ind]+((res_path[np.asarray((bend_ind-1,bend_ind+1))]-res_path[bend_ind])/saved_params['dx']*input_params['r_bend']).sum(axis = 0)
        res_path[bend_ind] = center_pnt+(res_path[bend_ind]-center_pnt)/np.linalg.norm((res_path[bend_ind]-center_pnt))*r_bend
        
        if input_params['truncated']:
            trunc_ind = np.unique(np.concatenate(([0,N[i]-1],bend_ind-1,bend_ind,bend_ind+1)).astype(int))
            # if np.any(np.any(np.unique(res_path[trunc_ind],axis = 0,return_counts = True)[1]>1)):
            #     print('double occupied')
            res_paths[i] = res_path[trunc_ind]

    V_res = np.sum(L*(np.sqrt(2)*input_params['a'])**2)
    V_ratio = V_res/saved_params['geom_offset'].volume
    print(f'Volume ratio: {np.round(V_ratio*100,1)}')

    success_rate = np.round(len(res_paths)/len(saved_params['res_nodes'])*100,2)
    print(f'Success rate: {success_rate}')

    packing_efficiency = np.round(np.sum(N)/saved_params['initial_unoccupied_nodes']*100,2)
    print(f'Packing Efficiency: {packing_efficiency}')

    return res_paths

def wrtie_res_paths(res_paths):
    

    save_dir = os.path.join(os.getcwd(),'res_paths')
    if os.path.exists(save_dir):
        rmtree(save_dir)
    os.mkdir(save_dir)

    for i,res in enumerate(res_paths):
        np.savetxt(os.path.join(save_dir,f'res{i}.csv'), X=res, fmt='%.8f', delimiter=',')

    # skip_ind = 2
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # # ax.scatter(saved_params['geom'].points[:,0],saved_params['geom'].points[:,1],saved_params['geom'].points[:,-1])
    # for path in res_paths[::skip_ind]:
    #     ax.plot(np.array(path)[:,0],np.array(path)[:,1],np.array(path)[:,2])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.grid(False)
    # ax.xaxis.pane.fill = False 
    # ax.yaxis.pane.fill = False 
    # ax.set_xticks(ax.get_xticks()[::2])
    # ax.set_yticks(ax.get_yticks()[::2])
    # ax.set_zticks(ax.get_zticks()[::2])

    return res_paths
