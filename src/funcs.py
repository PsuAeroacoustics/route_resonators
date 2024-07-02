#!/usr/bin/env python3
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import aerosandbox as asb
from collections import deque
from heapq import heappush,heappop
from itertools import count
import h5py
import os
#%%

stensile = np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])


def read_results_from_h5(case_dir):
    saved_params ={}
    with h5py.File(os.path.join(case_dir, 'saved_params.h5'), 'r') as f:
        for k,v in f.items():
            if isinstance(v[()], bytes):
                saved_params.update({k:v[()].decode()})
            else:
                saved_params.update({k:v[()]})
    return saved_params

class sample:
    def __init__(self,size = None, N_res = None,dx = None):
        self.size = size
        self.N_res = N_res
        self.occupied_nodes = deque()
        self.dx = dx

class resonator:
    def __init__(self,start_node = None,length = None,a = None):
        self.start_node =start_node
        self.length = length
        self.success = False
        self.a = a
    
    def get_path(self):
        if hasattr(self,'path'):
            path = np.flip(np.array([node.position for node in self.path]),axis = 0)
        # else:
        #     print("This resonator hasn't been routed")
            return path

class node(resonator):

    def __init__(self,f = None,parent = None, position = None,ind = None,count = None ,L = None, r = None):
        self.r = r
        self.parent = parent
        self.position = position
        self.ind = ind
        self.f = f
        self.count = count
        self.L = L

    def __lt__(self,other_node):
        return (self.f,self.count) < (other_node.f,other_node.count)

    def __gt__(self,other_node):
        return (self.f,self.count) > (other_node.f,other_node.count)

    def __eq__(self,other_node):
        return tuple(self.position) == tuple(other_node.position)
        # return np.linalg.norm(np.diff((self.position,other_node.position),axis = 0)) <= self.r+other_node.r

def build_blade_geom(saved_params):
    
    
    af = asb.Airfoil(saved_params['airfoil'])
    af.coordinates = af.repanel(n_points_per_side = int(saved_params['airfoil_points']/2)).coordinates*saved_params['c']
    af.coordinates[:,0] = -af.coordinates[:,0]
    af.coordinates[:,0] = af.coordinates[:,0]+0.25*saved_params['c']

    pnts_per_sections = len(af.coordinates)
    n_sections = saved_params['N_elements']+1

    blade_nodes = np.zeros((n_sections,pnts_per_sections,3))
    blade_nodes[:,:,1] = af.coordinates[:,0]
    blade_nodes[:,:,0] = np.expand_dims(saved_params['r_elem']*saved_params['R'],axis = -1)*np.ones(pnts_per_sections)
    blade_nodes[:,:,-1] = af.coordinates[:,-1]

    saved_params.update({'blade_nodes':blade_nodes})


def generate_domain(saved_params):

    num_Xsect = 32
    r_select = .5

    Xsect_ind = np.abs(r_select-saved_params['r']).argmin()
    Xsect_ind = slice(Xsect_ind,Xsect_ind+num_Xsect+1)

    blade_nodes = saved_params['blade_nodes'][Xsect_ind,:,:]
    pnts_per_Xsect = blade_nodes.shape[1]

    dx = np.array([np.min(saved_params['a'])/2,np.min(saved_params['a'])/2,np.min(saved_params['a'])/8])
    x_min,y_min,z_min = np.min(blade_nodes.reshape(np.product(blade_nodes.shape[:2]),3),axis = 0)
    x_max,y_max,z_max = np.max(blade_nodes.reshape(np.product(blade_nodes.shape[:2]),3),axis = 0)
    bounds = np.array([x_min,x_max,y_min,y_max,z_min,z_max])

    bound_range = bounds[1::2]- bounds[::2]
    x,y,z = [np.arange(bound_range[i]/dx[i])*dx[i]+bounds[::2][i] for i in range(3)]
    grid_coord = np.array(np.meshgrid(x,y,z))

    z_min = interp.griddata(points = blade_nodes[:,int(pnts_per_Xsect/2):,:-1].reshape(np.product(blade_nodes[:,int(pnts_per_Xsect/2):].shape[:-1]),2),values =blade_nodes[:,int(pnts_per_Xsect/2):,-1].flatten(),xi =(grid_coord[0],grid_coord[1]), fill_value=0,method = 'linear')
    z_max = interp.griddata(points = blade_nodes[:,:int(pnts_per_Xsect/2)+1,:-1].reshape(np.product(blade_nodes[:,:int(pnts_per_Xsect/2)+1].shape[:-1]),2),values =blade_nodes[:,:int(pnts_per_Xsect/2)+1,-1].flatten(),xi =(grid_coord[0],grid_coord[1]),fill_value=0,method = 'linear')

    # configures boolean grid that is True if a grid point is free and False if occupied
    grid = np.ones(grid_coord.shape[1:]).astype(bool)
    # sets the grid points that are located outside of the blade surface to False.
    grid[grid_coord[0]<=x_min] = False
    grid[grid_coord[0]>=x_max] = False
    grid[grid_coord[1]<=y_min] = False
    grid[grid_coord[1]>=y_max] = False
    grid[grid_coord[-1]<=z_min] = False
    grid[grid_coord[-1]>=z_max] = False

    saved_params.update({'x':x,'y':y,'z':z,'z_max':z_max,'z_min':z_min,'r_select':r_select,'num_Xsect':num_Xsect,'Xsect_ind':Xsect_ind,'dx':dx,'bounds':bounds,'grid_coord':grid_coord,'grid':grid})

def arange_resonators(saved_params):
    
    # number of unique resonators for each blade element
    N_res = len(saved_params['a'])
    # total number of resonators to route
    N_total = int(np.sum(N_res*saved_params['N'][saved_params['Xsect_ind']][:-1]))

    # minimum and maximum chord and spanwise extents of the resonator patch expressed as a percentage of the planform of the blade section (x_min,x_max,y_min,y_max)
    c_extents = np.array([.1,.35])*saved_params['c']
    c_bounds = (np.abs((saved_params['bounds'][3]-c_extents)-np.expand_dims(saved_params['y'],axis = -1))).argmin(axis = 0)
    
    x_ind = (np.random.rand(N_total)*np.diff(saved_params['bounds'][:2])/saved_params['dx'][0]).astype(int)
    y_ind = (np.random.rand(N_total)*abs(np.diff(c_bounds))+c_bounds.min()).astype(int)
    
    x_res = saved_params['x'][x_ind]
    y_res = saved_params['y'][y_ind]
    z_res = saved_params['z_max'][y_ind,x_ind,0]

    # type of resonator
    res_type = np.round(np.random.rand(len(z_res))*(N_res-1)).astype(int)

    saved_params.update({'N_total':N_total,'res_type':res_type,'x_res':x_res,'y_res':y_res,'z_res':z_res})

def route_resonators(saved_params):
   
    res_paths = []
    success = 0

    for i in range(saved_params['N_total']):
            # z_res_ind = np.squeeze(np.where(z_res[iter_column,iter_row]<z))[0]
        res = resonator(start_node=node(position = [saved_params['x_res'][i],saved_params['y_res'][i],saved_params['z_res'][i]],L = 0),length = saved_params['L'][saved_params['res_type'][i]],a=saved_params['a'][saved_params['res_type'][i]])
        route(res,saved_params)

        if res.success:
            res_paths.append(res.get_path())
            success+=1

    percent_fit = np.round(success/saved_params['N_total']*100)
    if percent_fit == 100:
        print(f"Woohoo all fit!")
    else:
            print(f"{percent_fit}% of resonators routed successfully")

    return res_paths

def route(res,saved_params):

    open_set = []
    closed_set = deque()
    cnt = count()
    
    heappush(open_set,res.start_node)

    while len(open_set) > 0:

        current_node = heappop(open_set)
        # current_node.r = resonator.r
        print(f"Current Node: {current_node.position} - Current Length: {current_node.L}")
        
        if current_node.L >= res.length:
            print('Path found!')
            res.path = trace_path(current_node,res.start_node)
            res.success = True
            break

        else:
            # calculates index of current node
            current_node.index = ((current_node.position-saved_params['bounds'][::2])/saved_params['dx']).round().astype(int)
            # finds indices of neighboring nodes
            # np.random.shuffle(stensile)
            neighbor_ind = current_node.index+stensile
            # removes indices that are on domain boundaries
            neighbor_ind = np.delete(neighbor_ind,np.where(saved_params['grid'].transpose(1,0,-1).shape == neighbor_ind)[0],axis = 0)
            # removes indices that are occupied or outside of the volume
            neighbor_ind = neighbor_ind[saved_params['grid'][neighbor_ind[:,1],neighbor_ind[:,0],neighbor_ind[:,-1]]]
            neighbors = saved_params['grid_coord'][:,neighbor_ind[:,1],neighbor_ind[:,0],neighbor_ind[:,-1]].T
            neighbors_dir = neighbor_ind-current_node.index
            saved_params['grid'][current_node.index[1],current_node.index[0],current_node.index[-1]] = False

            f = np.zeros(len(neighbors))
            if current_node ==  res.start_node:
                # defaults to growing resonators in -z direction
                f_ind = current_node.index[-1]-1!=neighbor_ind[:,-1]
                f[f_ind] = 1
            else:
                current_node.direction = current_node.index-current_node.parent.index
                bend_penalty_ind = np.sum(current_node.direction==neighbors_dir,axis = -1) != 3
                f[bend_penalty_ind] = 1

            for i,pos in enumerate(neighbors):
                temp_node = node(position=tuple(pos),parent=current_node,f = f[i], count = res.length-next(cnt))
                temp_node.L = current_node.L+saved_params['dx'][(neighbors_dir!=0)[i]]
                heappush(open_set,temp_node)

            closed_set.append(current_node) 

def trace_path(current_node,start_node):
    parent_set = deque()
    while current_node != start_node:
        current_node = current_node.parent
        parent_set.append(current_node) 
    return parent_set

