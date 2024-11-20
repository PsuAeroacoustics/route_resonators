#!/usr/bin/env python3
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import aerosandbox as asb
import h5py
import os
from random import randint
from shutil import rmtree
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


def build_blade_geom(saved_params):

    af = asb.Airfoil(saved_params['airfoil'])
    af.coordinates = af.repanel(n_points_per_side = int(saved_params['airfoil_points']/2)).coordinates*saved_params['c']
    # LE_ind,TE_ind = af.coordinates[:,0].argmin(),af.coordinates[:,0].argmax()
    af_s = af.coordinates[af.coordinates[...,-1]>0]
    af_p= af.coordinates[af.coordinates[...,-1]<0]
    t = af_s[:,-1].max() - af_p[:,-1][np.abs(af_s[af_s[:,-1].argmax()][0]-af_p[:,0]).argmin()]

    pnts_per_Xsect = len(af.coordinates)
    N_Xsect = saved_params['N_elements']+1

    blade_nodes = np.zeros((N_Xsect,pnts_per_Xsect,3))
    blade_nodes[:,:,0] = af.coordinates[:,0]
    blade_nodes[:,:,1] = np.expand_dims(saved_params['r_elem']*saved_params['R'],axis = -1)*np.ones(pnts_per_Xsect)
    blade_nodes[:,:,-1] = af.coordinates[:,-1]

    saved_params.update({'blade_nodes':blade_nodes,'pnts_per_Xsect':pnts_per_Xsect,'N_Xsect':N_Xsect,'t':t})


def initialize_grid(saved_params):

    dx = 1.025*np.max(2*saved_params['a'])

    x_min,y_min,z_min = np.min(np.min(saved_params['blade_nodes'],axis = 0),axis = 0)
    x_max,y_max,z_max = np.max(np.max(saved_params['blade_nodes'],axis = 0),axis = 0)

    Nx = np.ceil((x_max-x_min+2*dx)/dx)
    x = (np.arange(Nx)-np.floor(Nx/2))*dx+0.5*(x_max+x_min)

    Ny = np.ceil((y_max-y_min+2*dx)/dx)
    y = (np.arange(Ny)-np.floor(Ny/2))*dx+0.5*(y_max+y_min)

    Nz = np.ceil((z_max-z_min+2*dx)/dx)
    z = (np.arange(Nz)-np.floor(Nz/2))*dx+0.5*(z_max+z_min)
    grid_bounds = [x[0],x[-1],y[0],y[-1],z[0],z[-1]]

    grid_coord = np.array(np.meshgrid(x,y,z)).transpose(2,1,-1,0)
    grid = np.ones(grid_coord.shape[:-1]).astype(bool)

    x_mgrid,y_mgrid = np.array(np.meshgrid(x,y))

    z_p = interp.griddata(points = saved_params['blade_nodes'][:,int(saved_params['pnts_per_Xsect']/2):,:-1].reshape((np.prod(saved_params['blade_nodes'][:,int(saved_params['pnts_per_Xsect']/2):,:-1].shape[:-1]),2)),values =saved_params['blade_nodes'][:,int(saved_params['pnts_per_Xsect']/2):,-1].flatten(),xi= (x_mgrid,y_mgrid), fill_value=0,method = 'linear')
    z_s = interp.griddata(points = saved_params['blade_nodes'][:,:int(saved_params['pnts_per_Xsect']/2)+1,:-1].reshape((np.prod(saved_params['blade_nodes'][:,:int(saved_params['pnts_per_Xsect']/2)+1,:-1].shape[:-1]),2)),values =saved_params['blade_nodes'][:,:int(saved_params['pnts_per_Xsect']/2)+1,-1].flatten(),xi= (x_mgrid,y_mgrid), fill_value=0,method = 'linear')

    skin_thickness = 1e-3+saved_params['a'][0] 

    v_n = (np.array([-saved_params['blade_nodes'][...,-1],np.zeros(saved_params['blade_nodes'].shape[:2]),saved_params['blade_nodes'][...,0]])/np.linalg.norm(saved_params['blade_nodes'],axis = -1)).transpose(1,2,0)
    blade_nodes_offset = saved_params['blade_nodes']+v_n*skin_thickness

    blade_node_grad = np.gradient(saved_params['blade_nodes'],edge_order = 2,axis = 1)
    v_n = (np.array([-blade_node_grad[...,-1],np.zeros(blade_node_grad.shape[:2]),blade_node_grad[...,0]])/np.linalg.norm(blade_node_grad,axis = -1)).transpose(1,2,0)
    blade_nodes_offset = saved_params['blade_nodes']+v_n*skin_thickness

    z_p_offset = interp.griddata(points = blade_nodes_offset[:,int(saved_params['pnts_per_Xsect']/2):,:-1].reshape((np.prod(blade_nodes_offset[:,int(saved_params['pnts_per_Xsect']/2):,:-1].shape[:-1]),2)),values =blade_nodes_offset[:,int(saved_params['pnts_per_Xsect']/2):,-1].flatten(),xi= (x_mgrid,y_mgrid), fill_value=0,method = 'linear')
    z_s_offset = interp.griddata(points = blade_nodes_offset[:,:int(saved_params['pnts_per_Xsect']/2)+1,:-1].reshape((np.prod(blade_nodes_offset[:,:int(saved_params['pnts_per_Xsect']/2)+1,:-1].shape[:-1]),2)),values =blade_nodes_offset[:,:int(saved_params['pnts_per_Xsect']/2)+1,-1].flatten(),xi= (x_mgrid,y_mgrid), fill_value=0,method = 'linear')

    grid[grid_coord[...,-1]>=np.expand_dims(z_s_offset.T,axis = -1)] = False
    grid[grid_coord[...,-1]<=np.expand_dims(z_p_offset.T,axis = -1)] = False

    saved_params.update({'x':x,'y':y,'x_mgrid':x_mgrid,'y_mgrid':y_mgrid,'z':z,'z_s':z_s,'z_p_offset':z_p_offset,'z_s_offset':z_s_offset,'z_p':z_p,'dx':dx,'grid_coord':grid_coord,'grid':grid,'grid_bounds':grid_bounds,'skin_thickness':skin_thickness})


    # fig,ax = plt.subplots(1,1,figsize = (6.4,4.5))
    # ax.plot(saved_params['blade_nodes'][1,:,0].flatten()/saved_params['c'],saved_params['blade_nodes'][1,:,-1].flatten()/saved_params['c'])
    # ax.plot(blade_nodes_offset[1,:,0]/saved_params['c'],blade_nodes_offset[1,:,-1]/saved_params['c'])

    # # ax.plot(blade_nodes_offset[1,:,0].flatten()/saved_params['c'],blade_nodes_offset[1,:,-1].flatten()/saved_params['c'],linestyle = '--',color = 'grey')
    
    # ax.scatter(grid_coord[:,1,:,0]/saved_params['c'],grid_coord[:,1,:,-1]/saved_params['c'],c = 'black',s = 5)
    # ax.scatter(grid_coord[:,1,:,0][np.invert(grid[:,2])]/saved_params['c'],grid_coord[:,1,:,-1][np.invert(grid[:,2])]/saved_params['c'],c = 'red',s = 5)

    # ax.set_ylim([-0.1,0.1])
    # ax.set_xlabel('x/c')
    # ax.set_ylabel('z/c')
    # plt.grid()



def arange_resonators(saved_params, uniform = True):

    # number of unique resonators per blade element
    N_res = len(saved_params['a'])
    # total number of resonators to route
    N_total = (N_res*saved_params['N']).astype(int)
    c_extents = np.array([.1,.3])*saved_params['c']

    r_ind = (np.abs(saved_params['r_elem']*saved_params['R']-np.expand_dims(saved_params['y'],axis = -1))).argmin(axis = 0)
    c_ind = (np.abs(c_extents-np.expand_dims(saved_params['x'],axis = -1))).argmin(axis = 0)
    
    res_nodes = []
    starting_nodes = []
    starting_lengths = []
    res_types = []

    for i in range(np.sum(saved_params['filt_ind'])):

        grid_coord = np.array(list(map(lambda x:x[r_ind[:-1][saved_params['filt_ind']][i]:r_ind[1:][saved_params['filt_ind']][i],c_ind[0]:c_ind[1]][1:-1,1:-1],[saved_params['x_mgrid'],saved_params['y_mgrid'],saved_params['z_s']]))).T
        grid = np.ones(grid_coord.shape[:2],dtype=bool)

        if uniform:
            
            l_x = .2*saved_params['c']
            l_y = np.diff(saved_params['r_elem'][:2])[0]*saved_params['R']
            N_x = np.round(np.sqrt(N_total[i]*l_x/l_y))
            N_y = np.ceil(N_total[i]/N_x)

            skip_ind = np.round(np.array([grid_coord.shape[0]/N_x,grid_coord.shape[1]/N_y])).astype(int)
            res_nodes_temp = grid_coord[::skip_ind[0],::skip_ind[1]][:int(np.ceil(N_total[i]/(grid_coord.shape[1]/skip_ind[1])))]
            N_x,N_y = res_nodes_temp.shape[:2]
            res_nodes_temp = res_nodes_temp.reshape((np.product(res_nodes_temp.shape[:2]),3),order = 'F')

        else:
            
            r = np.sqrt(2)*1.025*np.max(saved_params['a'])
            active_pnts = []
            res_nodes_temp = []
            x0_ind = tuple((np.array(grid.shape)/2).astype(int))
            grid[x0_ind]= False
            active_pnt = grid_coord[x0_ind]
            active_pnts.append(grid_coord[x0_ind])

            while len(active_pnts)>=1 and len(res_nodes_temp)<N_total[i]:
                
                active_pnt = active_pnts.pop(randint(0,len(active_pnts)-1))
                # computes distance from the active point to the surrounding points
                active_pnt_dist = np.linalg.norm(grid_coord-active_pnt,axis = -1)
                test_pnt_ind = (active_pnt_dist<=2*r) & (active_pnt_dist >= r)
                test_pnts = grid_coord[test_pnt_ind][grid[test_pnt_ind]]
                
                for test_pnt_ind in range(len(test_pnts)):
                    test_pnt_dist = np.linalg.norm(grid_coord-test_pnts[test_pnt_ind],axis = -1)
                    if np.all(grid[test_pnt_dist<=r]):
                        grid[test_pnt_dist==0] = False
                        active_pnts.append(test_pnts[test_pnt_ind])
                        res_nodes_temp.append(test_pnts[test_pnt_ind])  
            
        res_nodes_temp = np.array(res_nodes_temp)[:N_total[i]]
        res_nodes_y = list(set(res_nodes_temp[:,1]))
        
        for y_iter in range(len(res_nodes_y)):
            nodes = np.sort(res_nodes_temp[res_nodes_temp[:,1]==res_nodes_y[y_iter]],axis=0, kind='stable')
            initial_depth = np.max((np.sum(saved_params['grid'][get_index(nodes[0],saved_params)[:2]]),len(nodes)))
            for node_iter,node in enumerate(nodes):
                node_ind = get_index(node,saved_params)
                N_nodes = initial_depth-node_iter%len(nodes)
                starting_nodes = saved_params['grid_coord'][node_ind[:2]][saved_params['grid'][node_ind[:2]]][::-1][:N_nodes]
                saved_params['grid'][get_index(starting_nodes,saved_params)] = False
                res_nodes.append(np.insert(starting_nodes,0, node,axis = 0))
        

        if N_res>1:
            if N_res<len(res_nodes_temp) and uniform:
                res_type = np.zeros(N_res,dtype=int)
                res_type[::2] = np.arange(np.ceil(N_res/2))
                res_type[1::2] = np.arange(np.floor(N_res/2))+np.ceil(N_res/2)
                res_type = np.array([np.roll(res_type,ii*int(N_res/2))[:int(N_y)] for ii in range(int(N_x))]).flatten()[:len(res_nodes_temp)] 
            else:
                res_type = np.random.permutation(np.tile(np.arange(N_res),int(np.ceil(len(res_nodes_temp)/N_res)))[:len(res_nodes_temp)])
        else:
            res_type = np.zeros(len(res_nodes),dtype = int)
        res_types.extend(res_type)      


    saved_params.update({'res_nodes':res_nodes,'res_types':res_types})


    # res_nodes = np.array(res_nodes)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(saved_params['blade_nodes'][:,:,0],saved_params['blade_nodes'][:,:,1],saved_params['blade_nodes'][:,:,-1] , alpha = .1,color = 'gray')
    # # ax.plot_surface(saved_params['x_mgrid'],saved_params['y_mgrid'],saved_params['z_p'],linewidth = 10 , alpha = .5,color = 'gray')
    # # ax.scatter(grid_coord.T[0],grid_coord.T[1],grid_coord.T[2],alpha=.25)
    # ax.scatter(res_nodes[:,0],res_nodes[:,1],res_nodes[:,2])
    # ax.set_xlim(-0.05,0.05)
    # ax.set_ylim(0.2,0.3)
    # ax.set_zlim(-0.025,0.025)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.grid(False)
    # ax.xaxis.pane.fill = False 
    # ax.yaxis.pane.fill = False 
    # ax.set_xticks(ax.get_xticks()[::2])
    # ax.set_yticks(ax.get_yticks()[::2])
    # ax.set_zticks(ax.get_zticks()[::2])


def route(starting_node,total_length,saved_params):

    # starting length determined by the distance between the starting point and the nearest point inside the geometry 
    L = np.abs(starting_node[-1]-starting_node[0])[-1]
    # starting direction downwards in z
    active_direction = np.array([1,0,0])
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
            closed_set = closed_set[:trim_ind]
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
                if len(pnts)>0 and np.all(saved_params['grid'][get_index(pnts[:2],saved_params)]):
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
                
                    if active_direction_ind:
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
        saved_params['grid'][get_index(np.array(closed_set),saved_params)] = True
        closed_set = []
    return closed_set

def route_resonators(saved_params):
   
    res_paths = []
    if len(saved_params['L'])>1:
        route_order_ind = np.array(saved_params['res_types']).argsort(kind = 'mergesort')[::-1]
    else:
        route_order_ind = np.random.permutation(np.arange(len(saved_params['res_nodes'])))

    for i in range(len(saved_params['res_nodes'])):
        res_paths.append(route(starting_node = saved_params['res_nodes'][route_order_ind[i]],total_length = saved_params['L'][saved_params['res_types'][route_order_ind[i]]],saved_params = saved_params))
    
    N_routed_ind = np.array([len(i) for i in res_paths])!=0
    N_routed = np.sum(N_routed_ind)
    # route_order_ind = route_order_ind[np.invert(N_routed_ind)]
    success_rate = np.round(N_routed/len(saved_params['res_nodes'])*100,2)
    res_paths = list(filter(lambda x: x, res_paths))

    return res_paths

def wrtie_res_paths(res_paths):
    

    save_dir = os.path.join(os.getcwd(),'res_paths')
    if os.path.exists(save_dir):
        rmtree(save_dir)
    os.mkdir(save_dir)

    for i,res in enumerate(res_paths):
        np.savetxt(os.path.join(save_dir,f'res{i}.csv'), X=res, fmt='%.8f', delimiter=',')

    # N_elems = 1
    # resorted_ind = route_order_ind.argsort(kind = 'mergesort')
    
    # c = np.random.choice(np.array(list(mcolors.CSS4_COLORS.keys())),len(saved_params['a']))
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(saved_params['blade_nodes'][:,:,0],saved_params['blade_nodes'][:,:,1],saved_params['blade_nodes'][:,:,-1] , alpha = .2,color = 'gray')
    # # ax.plot_surface(saved_params['x_m  grid'],saved_params['y_mgrid'],saved_params['z_p'],linewidth = 10 , alpha = .5,color = 'gray')
    # # for i in range(int(N_elems*saved_params['N'][0]*len(saved_params['a']))):
    # # for i in range(300):
    # #     ax.plot(np.array(res_paths[i])[:,0],np.array(res_paths[i])[:,1],np.array(res_paths[i])[:,2])
    # for i in range(int(N_elems*saved_params['N'][0]*len(saved_params['a']))):
    #     if len(res_paths[resorted_ind[i]])>0:
    #     # ax.plot(np.array(res_paths[i])[:,0],np.array(res_paths[i])[:,1],np.array(res_paths[i])[:,2])
    #         # ax.plot(np.array(res_paths[resorted_ind[i]])[:,0],np.array(res_paths[resorted_ind[i]])[:,1],np.array(res_paths[resorted_ind[i]])[:,2],c = c[saved_params['res_types'][resorted_ind[i]]])
    #         ax.plot(np.array(res_paths[resorted_ind[i]])[:,0],np.array(res_paths[resorted_ind[i]])[:,1],np.array(res_paths[resorted_ind[i]])[:,2])

    # ax.set_xlim(-0.05,0.05)
    # ax.set_ylim(0.2,0.3)
    # ax.set_zlim(-0.025,0.025)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.grid(False)
    # ax.xaxis.pane.fill = False 
    # ax.yaxis.pane.fill = False 
    # ax.set_xticks(ax.get_xticks()[::2])
    # ax.set_yticks(ax.get_yticks()[::2])
    # ax.set_zticks(ax.get_zticks()[::2])

    # return res_paths
