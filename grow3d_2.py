import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from heapq import heappush,heappop,heappushpop
from itertools import count
import time
from AnalyzeDegenGeom import AnalyzeDegenGeom
import os
import scipy.interpolate as interp
import h5py

#%%

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


def trace_path(current_node,start_node):
    parent_set = deque()
    while current_node != start_node:
        current_node = current_node.parent
        parent_set.append(current_node) 
    return parent_set

def get_index(node):
    return ((node.position-bounds[::2])/dx).round().astype(int)


def route(res):

    open_set = []
    closed_set = deque()
    cnt = count()
    
    heappush(open_set,res.start_node)

    while len(open_set) > 0:

        current_node = heappop(open_set)
        # current_node.r = resonator.r
        print(f'Current Node: {current_node.position} - Current Length: {current_node.L}')
        
        if current_node.L >= res.length:
            print('Path found!')
            res.path = trace_path(current_node,res.start_node)
            # sample.occupied_nodes.extend(res.path)
            res.success = True
            break

        else:
            current_node.index = get_index(current_node)
            neighbor_ind = get_index(current_node)+stensile
            neighbor_ind = neighbor_ind[grid[neighbor_ind[:,1],neighbor_ind[:,0],neighbor_ind[:,-1]]]
            neighbors = grid_coord.transpose(1,2,3,0)[neighbor_ind[:,1],neighbor_ind[:,0],neighbor_ind[:,-1]]
            neighbors_dir = neighbor_ind-current_node.index
            
            f = np.zeros(len(neighbors))
            if current_node ==  res.start_node:
                # defaults to growing resonators in -z direction
                f[:-1] = 1
            else:
                current_node.direction = current_node.index-current_node.parent.index
                bend_penalty_ind = np.sum(current_node.direction==neighbors_dir,axis = -1) != 3
                f[bend_penalty_ind] = 1
            
            for i,pos in enumerate(neighbors):
                temp_node = node(position=tuple(pos),parent=current_node,f = f[i], count = res.length-next(cnt))
                temp_node.L = current_node.L+dx[(neighbors_dir!=0)[i]]

                # ind = np.squeeze(np.where([temp_node == existing_node for existing_node in open_set]))
                # if ind.size !=  0:
                #     if temp_node.f < open_set[ind].f:
                #         open_set[ind].f = temp_node.f
                #         # open_set[ind].remaining_dist = abs(L-L_current-L_remainder)
                #         open_set[ind].parent = temp_node.parent
                # else:
                heappush(open_set,temp_node)

            # print(f'{current_node.position} - {neighbors}')

            closed_set.append(current_node) 

    
#%% imports information about the resonator geometry and spanfise distribution
data_dir = '/home/dfw5266/codes/GitHub/PsuAeroacoustics/hart_ii_python/'
data_fname = 'geom_mdof_3_2'
res_params = {}
with h5py.File(os.path.join(data_dir,data_fname,'res_params.h5'),'r') as f:
    for k,v in f.items():
        # res_params['res_opt] = [# of each resonator, radius of cavities, length of cavities] - for each resonator
        res_params = {**res_params,**{k:v[()]}}

#%% import blade degenerate geometry from OpenVSP

af_data_dir = os.path.dirname(__file__)
af_fname = 'bo105_hart_DegenGeom.csv'
dataSorted, indHeader = AnalyzeDegenGeom(os.path.join(af_data_dir,af_fname))
surfNodes = np.float64(dataSorted['Component 1']['SURFACE_NODE'][1:, :3])
nXsect = int(indHeader[0][1][1])
pntsPerXsect = int(indHeader[0][1][2])

#%%
# spanwise location to route
r_select = 0.75
# number of selected spanwise sections to use as volume
nXsect_select = 1
# spatial resolution dx,dy,dz
dx = np.min(res_params['a'])/2*np.ones(3)

#%% Processes airfoil data and determines domain bounds

surfNodes = surfNodes.reshape((pntsPerXsect,nXsect,3) ,order = 'F')
R = np.mean(surfNodes[:,-1,1])
Xsect_select = int(r_select*nXsect)
surfNodes_select = surfNodes[:,Xsect_select:Xsect_select+nXsect_select+1]

x_min,x_max,y_min,y_max,z_min,z_max = np.array(list(map(lambda x:  [np.min(x),np.max(x)], surfNodes_select.T))).flatten()
bounds = np.array([x_min-2*dx[0],x_max+2*dx[0],y_min-2*dx[1],y_max+2*dx[1],z_min-2*dx[2],z_max+2*dx[2]])

#%%
bound_range = bounds[1::2]- bounds[::2]
x,y,z = [np.arange(bound_range[i]/dx[i]+1)*dx[i]+bounds[::2][i] for i in range(3)]
grid_coord = np.array(np.meshgrid(x,y,z))

#%%
LE_ind = int(np.where(surfNodes[:,0,0]==surfNodes[:,0,0].min())[0])
p_surf = surfNodes[:LE_ind+1].reshape(np.product(surfNodes[LE_ind:].shape[:2]),3,order = 'f')
s_surf = surfNodes[LE_ind:].reshape(np.product(surfNodes[LE_ind:].shape[:2]),3,order = 'f')
z_min = interp.griddata(points = p_surf[:,:-1],values =p_surf[:,-1],xi =(grid_coord[0],grid_coord[1]), fill_value=0,method = 'linear')
z_max = interp.griddata(points = s_surf[:,:-1],values =s_surf[:,-1],xi =(grid_coord[0],grid_coord[1]),fill_value=0,method = 'linear')

# configures boolean grid that is True if a grid point is free and False if occupied
grid = np.ones(grid_coord.shape[1:]).astype(bool)
# sets the grid points that are located outside of the blade surface to False.
grid[grid_coord[0]<=x_min] = False
grid[grid_coord[0]>=x_max] = False
grid[grid_coord[1]<=y_min] = False
grid[grid_coord[1]>=y_max] = False
grid[grid_coord[-1]<=z_min] = False
grid[grid_coord[-1]>=z_max] = False

#%% determines arangement of resonators

# number of each type of unique resonators comprising an impedance patch
N = res_params['N'][int(r_select*len(res_params['N']))]
# number of unique resonators
N_res = len(res_params['a'])
# total number of resonators comprising the patch
N_total = int(N*N_res)

# minimum and maximum chord and spanwise extents of the resonator patch expressed as a percentage of the planform of the blade section (x_min,x_max,y_min,y_max)
res_extents = [.1,.35,.1,.9]

# number of resonators in the spanwise (y) direction
N_y_res =int(np.floor((y_max-y_min)*(res_extents[3]-res_extents[2])/(np.max(res_params['a']))))
# center-to-center spanwise spacing of resonators
N_y_spacing = (y_max-y_min)*(res_extents[3]-res_extents[2])/(N_y_res+1)
# number of indices to skip to achieve the desired spanwise resonator spacing
y_ind_skip = int(np.round(N_y_spacing/dx[1]))
# starting index corresponding to the first resonator in the spanwise direction
start_y_ind = np.where((y-(y_min+(y_max-y_min)*res_extents[2]))==abs(y-(y_min+(y_max-y_min)*res_extents[2])).min())[0][0]
# y-coordinates of resonator cavities
y_res  = y[start_y_ind::y_ind_skip][:N_y_res]
y_res_ind = ((y_res-bounds[2])/dx[1]).round().astype(int)

# number of resonators in the chordwise (x) direction
N_x_res = int(np.round(N_total/N_y_res))
# center-to-center spacing of spanwise resonators
N_x_spacing = (x_max-x_min)*(res_extents[1]-res_extents[0])/(N_x_res+1)
x_ind_skip = int(np.round(N_x_spacing/dx[0]))
start_x_ind = np.where(abs(x-(x_max-(x_max-x_min)*res_extents[0]))==abs(x-(x_max-(x_max-x_min)*res_extents[0])).min())[0][0]
# x-coordinates of resonator cavities
x_res = x[:start_x_ind][::-x_ind_skip][:N_x_res][::-1]
x_res_ind = ((x_res-bounds[0])/dx[0]).round().astype(int)

# x_res,y_res = np.meshgrid(x_res,y_res)
z_res = z_max[y_res_ind][:,x_res_ind][:,:,0]
z_res_ind = ((z_res-bounds[4])/dx[2]).round().astype(int)
z_res = z[z_res_ind]

# type of resonator
res_type = np.round(np.random.rand(N_y_res,N_x_res)*(N_res-1)).astype(int)

# finite difference stencile
stensile = np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])

res = {}
for x_iter in range(N_x_res):
    for y_iter in range(N_y_res):
        # z_res_ind = np.squeeze(np.where(z_res[iter_column,iter_row]<z))[0]
        res_temp = resonator(start_node=node(position = [x_res[x_iter],y_res[y_iter],z_res[y_iter,x_iter]],L = 0),length = res_params['L'][res_type[y_iter,x_iter]],a=res_params['a'][res_type[y_iter,x_iter]])
        res = {**res,**{f'res{x_iter*N_x_res+y_iter}':res_temp}}

[route(v) for k,v in res.items()]

success = 0
for k,v in res.items():
    if v.success:
        success+=1

paths = [v.get_path() for k,v in res.items() if len(v.path)>1]


#%%

fig = plt.figure()
ax = plt.axes(projection='3d') 
# ax.set_box_aspect(((np.max(surfNodes[:,:,0])-np.min(surfNodes[:,:,0]))/(np.max(surfNodes[:,:,0])-np.min(surfNodes[:,:,0])),(np.max(surfNodes[:,:,1])-np.min(surfNodes[:,:,1]))/(np.max(surfNodes[:,:,0])-np.min(surfNodes[:,:,0])),(np.max(surfNodes[:,:,-1])-np.min(surfNodes[:,:,-1]))/(np.max(surfNodes[:,:,0])-np.min(surfNodes[:,:,0]))))
for path in paths:
     ax.plot(path[:,0],path[:,1],path[:,2],linewidth = 5)

ax.plot_surface(surfNodes[:,:,0],surfNodes[:,:,1],surfNodes[:,:,2],alpha = .2)
# ax.set_xlim(liner.size[0],liner.size[1])
# ax.set_ylim(liner.size[2],liner.size[3])
# ax.set_zlim(liner.size[4],liner.size[5])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.set_xlim(bounds[:2])
# ax.set_ylim(bounds[2:4])
# ax.set_zlim(bounds[4:])
plt.grid()
plt.savefig('res.png')

#%%

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(s_surf[:,0],s_surf[:,1],s_surf[:,2])
ax.scatter3D(p_surf[:,0],p_surf[:,1],p_surf[:,2])
ax.scatter3D(grid_coord[0],grid_coord[1],p_surf_spline)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_ylim([r_select*R-.1*(r_select*R),r_select*R+.1*(r_select*R)])
plt.grid()
plt.savefig('res_geom.png')