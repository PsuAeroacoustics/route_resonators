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
from random import choice,randint
#%%

stensile_1 = np.array([(0,0,-1),(1,0,0),(0,1,0),(-1,0,0),(0,-1,0),(0,0,1)])
stensile_2 = np.array([(0,0,-2),(2,0,0),(0,2,0),(-2,0,0),(0,-2,0),(0,0,2)])
rng = np.random.default_rng()

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
    
    # def get_path(self):
    #     if hasattr(self,'path'):
    #         path = np.flip(np.array([node.position for node in self.path]),axis = 0)
    #     # else:
    #     #     print("This resonator hasn't been routed")
    #         return path

class node(resonator):

    def __init__(self,parent = None, position = None,ind = None,count = None ,L = None, r = None,index = None,f = 0,direction = [0,0,-1]):
        self.r = r
        self.parent = parent
        self.position = position
        self.ind = ind
        self.f = f
        self.count = count
        self.L = L
        self.direction = direction
        self.index = index

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
    # LE_ind,TE_ind = af.coordinates[:,0].argmin(),af.coordinates[:,0].argmax()

    pnts_per_Xsect = len(af.coordinates)
    N_Xsect = saved_params['N_elements']+1

    blade_nodes = np.zeros((N_Xsect,pnts_per_Xsect,3))
    blade_nodes[:,:,0] = af.coordinates[:,0]
    blade_nodes[:,:,1] = np.expand_dims(saved_params['r_elem']*saved_params['R'],axis = -1)*np.ones(pnts_per_Xsect)
    blade_nodes[:,:,-1] = af.coordinates[:,-1]

    saved_params.update({'blade_nodes':blade_nodes,'pnts_per_Xsect':pnts_per_Xsect,'N_Xsect':N_Xsect})

    # af = asb.Airfoil(saved_params['airfoil'])
    # af.coordinates = af.repanel(n_points_per_side = int(saved_params['airfoil_points']/2)).coordinates*(saved_params['c']-2*saved_params['a'][0])

    # interior_blade_nodes = np.zeros((n_sections,pnts_per_sections,3))
    # interior_blade_nodes[:,:,0] = af.coordinates[:,0]
    # interior_blade_nodes[:,:,1] = np.expand_dims(saved_params['r_elem']*saved_params['R'],axis = -1)*np.ones(pnts_per_sections)
    # interior_blade_nodes[:,:,-1] = af.coordinates[:,-1]


    # th_tw = -6*np.pi/180
    # th = th_tw*saved_params['r_elem']

    # def get_dcm(th):
    #     dcm = np.array([[np.ones(len(th)),np.zeros(len(th)),np.zeros(len(th))],
    #             [np.zeros(len(th)),np.cos(th),np.sin(th)],
    #             [np.zeros(len(th)),np.sin(th),np.cos(th)]]).squeeze()
    #     return dcm
    
    # dcm_th_tw = get_dcm(th)
    # dcm_th0 = get_dcm([th_tw])

    # blade_nodes = blade_nodes@dcm_th_tw.T

    # fig,ax = plt.subplots(1,1,figsize = (6.4,4.5))
    # ax.plot(blade_nodes[0,:,0],blade_nodes[0,:,-1])
    # ax.plot(interior_blade_nodes[0,:,0],interior_blade_nodes[0,:,-1])

    # # ax.plot(blade_nodes[0,:,1],blade_nodes[0,:,2])
    # # ax.plot(blade_nodes_tw[0,:,1],blade_nodes_tw[0,:,2])
    # # ax.set_xlabel('y')
    # # ax.set_ylabel('z')
    # ax.set_xlim([0,0.015])
    # ax.set_ylim([-0.005,0.005])
    # plt.grid()
    # plt.savefig('af_xsect.png',format = 'png')
    # plt.close()
    # saved_params.update({'blade_nodes':blade_nodes})


def initialize_grid(saved_params):

    dx = 1.025*np.min(saved_params['a'])

    # tan = np.gradient(saved_params['blade_nodes'],axis = 1,edge_order=2).T
    # norm = np.array((-tan[-1],tan[0]))/np.linalg.norm((tan[0],tan[-1]),axis = 0)
    # norm = np.insert(norm,1,np.zeros(norm.shape[1:]),axis = 0)
    # blade_nodes_offset = saved_params['blade_nodes']+norm.T*dx

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

    # z_p_offset = interp.griddata(points = blade_nodes_offset[:,int(saved_params['pnts_per_Xsect']/2):,:-1].reshape((np.prod(blade_nodes_offset[:,int(saved_params['pnts_per_Xsect']/2):,:-1].shape[:-1]),2)),values =blade_nodes_offset[:,int(saved_params['pnts_per_Xsect']/2):,-1].flatten(),xi= (x_mgrid,y_mgrid), fill_value=0,method = 'linear')
    # z_s_offset = interp.griddata(points = blade_nodes_offset[:,:int(saved_params['pnts_per_Xsect']/2)+1,:-1].reshape((np.prod(blade_nodes_offset[:,:int(saved_params['pnts_per_Xsect']/2)+1,:-1].shape[:-1]),2)),values =blade_nodes_offset[:,:int(saved_params['pnts_per_Xsect']/2)+1,-1].flatten(),xi= (x_mgrid,y_mgrid), fill_value=0,method = 'linear')

    grid[grid_coord[...,-1]>=np.expand_dims(z_s.T,axis = -1)] = False
    grid[grid_coord[...,-1]<=np.expand_dims(z_p.T,axis = -1)] = False

    saved_params.update({'x':x,'y':y,'x_mgrid':x_mgrid,'y_mgrid':y_mgrid,'z':z,'z_s':z_s,'z_p':z_p,'dx':dx,'grid_coord':grid_coord,'grid':grid,'grid_bounds':grid_bounds})



    # fig,ax = plt.subplots(1,1,figsize = (6.4,4.5))
    # ax.plot(saved_params['blade_nodes'][1,:,0].flatten()/saved_params['c'],saved_params['blade_nodes'][1,:,-1].flatten()/saved_params['c'])
    # # ax.plot(blade_nodes_offset[1,:,0].flatten()/saved_params['c'],blade_nodes_offset[1,:,-1].flatten()/saved_params['c'],linestyle = '--',color = 'grey')
    
    # ax.scatter(grid_coord[:,1,:,0]/saved_params['c'],grid_coord[:,1,:,-1]/saved_params['c'],c = 'black',s = 5)
    # ax.scatter(grid_coord[:,1,:,0][np.invert(grid[:,2])]/saved_params['c'],grid_coord[:,1,:,-1][np.invert(grid[:,2])]/saved_params['c'],c = 'red',s = 5)

    # ax.set_ylim([-0.2,0.2])
    # ax.set_xlabel('x/c')
    # ax.set_ylabel('z/c')
    # plt.grid()



def arange_resonators(saved_params):
    
    # number of unique resonators for each blade element
    N_res = len(saved_params['a'])
    # total number of resonators to route
    N_total = np.sum(N_res*saved_params['N'][saved_params['Xsect_ind']][:-1]).astype(int)

    # minimum and maximum chord and spanwise extents of the resonator patch expressed as a percentage of the planform of the blade section (x_min,x_max,y_min,y_max)
    c_extents = np.array([.2,.6])*saved_params['c']
    c_bounds = (np.abs((saved_params['bounds'][3]-c_extents)-np.expand_dims(saved_params['y'],axis = -1))).argmin(axis = 0)
    
    x_ind = (np.random.rand(N_total)*np.diff(saved_params['bounds'][:2])/saved_params['dx'][0]).astype(int)
    y_ind = (np.random.rand(N_total)*abs(np.diff(c_bounds))+c_bounds.min()).astype(int)
    
    x_res = saved_params['x'][x_ind]
    y_res = saved_params['y'][y_ind]
    z_res = saved_params['z_max'][y_ind,x_ind,0]

    # type of resonator
    res_type = np.round(np.random.rand(len(z_res))*(N_res-1)).astype(int)

    saved_params.update({'N_total':N_total,'res_type':res_type,'x_res':x_res,'y_res':y_res,'z_res':z_res})

def poisson_disc(saved_params, uniform = True):

    # number of unique resonators per blade element
    N_res = len(saved_params['a'])
    # total number of resonators to route
    N_total = (N_res*saved_params['N']).astype(int)
    r = np.sqrt(2)*saved_params['dx']
    c_extents = np.array([.15,.4])*saved_params['c']

    r_ind = (np.abs(saved_params['r_elem']*saved_params['R']-np.expand_dims(saved_params['y'],axis = -1))).argmin(axis = 0)
    c_ind = (np.abs(c_extents-np.expand_dims(saved_params['x'],axis = -1))).argmin(axis = 0)
    
    res_nodes = []
    starting_nodes = []
    starting_lengths = []
    res_types = []

    for i in range(len(saved_params['r'][saved_params['filt_ind']])-1):
        grid_coord = np.array(list(map(lambda x:x[r_ind[i]+1:r_ind[i+1]-1,c_ind[0]:c_ind[1]],[saved_params['x_mgrid'],saved_params['y_mgrid'],saved_params['z_s']]))).transpose(1,2,0)
        grid = np.ones(grid_coord.shape[:2],dtype=bool)

        if uniform:
            # N_y = np.round((saved_params['y'][r_ind[i+1]-1]-saved_params['y'][r_ind[i]])/(2*saved_params['dx']))
            # y_interval = int(np.round(len(saved_params['y'][r_ind[i]+1:r_ind[i+1]-1])/N_x))
            # N_x = np.round(((saved_params['x'][c_ind[1]]-saved_params['x'][c_ind[0]]))/(2*saved_params['dx']))
            # x_interval = int(np.round(len(saved_params['x'][c_ind[0]:c_ind[1]])/N_x))

            res_nodes_temp = grid_coord[::2,::2].reshape(np.product(grid_coord[::2,::2].shape[:2]),3,order = 'F')[:N_total[i]]
            res_nodes.extend(res_nodes_temp)

        else:
            active_pnts = []
            res_nodes_temp = []
            x0_ind = tuple((np.array(grid.shape)/2).astype(int))
            grid[x0_ind]= False
            active_pnt = grid_coord[x0_ind]
            active_pnts.append(grid_coord[x0_ind])
            res_nodes_temp.append(grid_coord[x0_ind])  

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
            res_nodes.extend(res_nodes_temp)

        starting_nodes_temp = np.copy(res_nodes_temp)
        starting_nodes_temp[:,-1] = np.array([saved_params['z'][(res_nodes_temp[i,-1] >=saved_params['z'])][-1] for i in range(len(res_nodes_temp))])
        starting_nodes.extend(starting_nodes_temp)

        starting_lengths.extend(res_nodes_temp[:,-1]-starting_nodes_temp[:,-1])

        res_types_temp = np.repeat(np.arange(N_res),int(N_total[i]/N_res))
        np.random.shuffle(res_types_temp)
        res_types.extend(res_types_temp)

    starting_nodes_ind = np.round((np.array(starting_nodes)-saved_params['grid_bounds'][::2])/saved_params['dx']).astype(int)
    saved_params['grid'][tuple(starting_nodes_ind.T)] = False

    saved_params.update({'res_nodes':res_nodes,'starting_nodes':starting_nodes,'starting_lengths':starting_lengths,'res_types':res_types})


    # res_nodes = np.array(res_nodes)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(saved_params['blade_nodes'][:,:,0],saved_params['blade_nodes'][:,:,1],saved_params['blade_nodes'][:,:,-1] , alpha = .2,color = 'gray')
    # # ax.plot_surface(saved_params['x_mgrid'],saved_params['y_mgrid'],saved_params['z_p'],linewidth = 10 , alpha = .5,color = 'gray')
    # ax.scatter(grid_coord.T[0],grid_coord.T[1],grid_coord.T[2],alpha=.25)
    # ax.scatter(res_nodes[:,0],res_nodes[:,1],res_nodes[:,2],c = 'r')
    # ax.set_ylim(0.055,0.065)
    # ax.set_zlim(0,0.005)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.grid(False)
    # ax.xaxis.pane.fill = False 
    # ax.yaxis.pane.fill = False 
    # ax.set_xticks(ax.get_xticks()[::2])
    # ax.set_yticks(ax.get_yticks()[::2])
    # ax.set_zticks(ax.get_zticks()[::2])


def route_resonators(saved_params):
   
    res_paths = []
    success = 0

    # for i in range(len(saved_params['res_nodes'])):
    for i in range(int(len(saved_params['a'])*saved_params['N'][0])):

            # z_res_ind = np.squeeze(np.where(z_res[iter_column,iter_row]<z))[0]
        res = resonator(start_node=node(position = saved_params['starting_nodes'][i],L = saved_params['starting_lengths'][i]),length = saved_params['L'][saved_params['res_types'][i]],a=saved_params['a'][saved_params['res_types'][i]])
        route(res,saved_params,shuffle = False)

        if res.success:
            res_paths.append(res.path)
            success+=1

    # percent_fit = np.round(success/saved_params['N_total']*100)
    # if percent_fit == 100:
    #     print(f"Woohoo all fit!")
    # else:
    #         print(f"{percent_fit}% of resonators routed successfully")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(saved_params['blade_nodes'][:,:,0],saved_params['blade_nodes'][:,:,1],saved_params['blade_nodes'][:,:,-1] , alpha = .2,color = 'gray')
    # ax.plot_surface(saved_params['x_mgrid'],saved_params['y_mgrid'],saved_params['z_p'],linewidth = 10 , alpha = .5,color = 'gray')
    for i in range(len(res_paths)):
        ax.scatter(np.array(res_paths[i]).squeeze()[0,0],np.array(res_paths[i]).squeeze()[0,1],np.array(res_paths[i]).squeeze()[0,2])
        ax.plot(np.array(res_paths[i]).squeeze()[:,0],np.array(res_paths[i]).squeeze()[:,1],np.array(res_paths[i]).squeeze()[:,2])
    ax.set_ylim(0.055,0.065)
    ax.set_zlim(0,0.005)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid(False)
    ax.xaxis.pane.fill = False 
    ax.yaxis.pane.fill = False 
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_zticks(ax.get_zticks()[::2])

    return res_paths

def route(res,saved_params,shuffle = True):

    open_set = []
    closed_set = []
    cnt = count()
    
    open_set.append(res.start_node)
    # closed_set.append(res.start_node)
    # temp_node = res.start_node.position+np.array(res.start_node.direction)*saved_params['dx']
    # temp_node = node(position=temp_node,parent=res.start_node,f = 0,L = res.start_node.L+saved_params['dx'], direction=res.start_node.direction)
    # open_set.append(temp_node)


    while len(open_set) > 0:

        current_node = open_set.pop()
        # current_node.r = resonator.r
        print(f"Current Node: {current_node.position} - Current Length: {current_node.L}")
        
        if current_node.L >= res.length:
            print('Path found!')
            res.path = list(map(lambda x:x.position, closed_set))
            res.success = True
            break

        else:
            current_node.index = np.round((current_node.position-saved_params['grid_bounds'][::2])/saved_params['dx']).astype(int)

                # np.random.shuffle(stensile_1)
            neighbor_ind = np.array([current_node.index+stensile_1,current_node.index+stensile_2]).transpose(-1,1,0)
            
            # remove indices that are outside of the domain
            out_of_bounds_pnts = ((saved_params['grid'].shape <= neighbor_ind.T) | (0 > neighbor_ind.T))
            if np.any(out_of_bounds_pnts):
                neighbor_ind = np.delete(neighbor_ind,np.where(out_of_bounds_pnts)[1],axis = 1)

            # removes indices that are occupied or outside of the volume
            neighbor_ind = np.delete(neighbor_ind,np.where(np.invert(saved_params['grid'][tuple(neighbor_ind)]))[0],axis = 1)

        if neighbor_ind.shape[1]!=0:
            if shuffle:
                rng.shuffle(neighbor_ind,axis = 1)

            neighbors = saved_params['grid_coord'][tuple(neighbor_ind)]
            neighbors_dir = neighbor_ind[...,0].T-current_node.index

            current_dit_ind = np.all(current_node.direction==neighbors_dir,axis = -1)
            if current_node==res.start_node:
                neighbors = neighbors[current_dit_ind]
                neighbors_dir = neighbors_dir[current_dit_ind]
                f = [0]
            else:
                f = np.ones(len(neighbors))
                f[current_dit_ind] = 0
                f_sorted_ind = f.argsort()[::-1]
                f = f[f_sorted_ind]
                neighbors = neighbors[f_sorted_ind]
                neighbors_dir = neighbors_dir[f_sorted_ind]

            for i in range(len(neighbors)):
                temp_node = node(position=neighbors[i,0],parent=current_node,f = f[i],L = current_node.L+saved_params['dx'],direction=neighbors_dir[i])
                if f[i] !=0:
                    temp_node = node(position=neighbors[i,1],parent=temp_node,f = f[i],L = current_node.L+2*saved_params['dx'], direction=neighbors_dir[i])
                open_set.append(temp_node)
            
            if current_node.f !=0:
                current_node.parent.index = np.round((current_node.parent.position-saved_params['grid_bounds'][::2])/saved_params['dx']).astype(int)
                saved_params['grid'][tuple(current_node.parent.index)]=False
                closed_set.append(current_node.parent)

            saved_params['grid'][tuple(current_node.index)]=False
            closed_set.append(current_node)
        
        else:
            if len(open_set)>0:
                if closed_set[-1]!=open_set[-1].parent.parent:
                    while closed_set[-1]!=open_set[-1].parent.parent:
                        for i in range(int(closed_set[-1].f+1)):
                            remove_node = closed_set.pop()
                            remove_node.index = np.round((remove_node.position-saved_params['grid_bounds'][::2])/saved_params['dx']).astype(int)
                            saved_params['grid'][tuple(remove_node.index)]=True
