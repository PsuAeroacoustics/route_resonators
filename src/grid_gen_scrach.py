import aerosandbox as asb
import numpy as np
import matplotlib.pyplot as plt

#%%

R = 0.1905
c = R/12.5
dx = 0.0004
af = asb.Airfoil('naca0012')
af.coordinates = (af.repanel(n_points_per_side = int(c/dx),spacing_function_per_side=np.linspace).coordinates*c).T

offset = np.arange(1,2)*dx
tan_vect = np.gradient(af.coordinates,axis = -1,edge_order=2)
# tan_vect = (tan_vect.T/np.linalg.norm(tan_vect,axis = -1)).T
norm_vect = np.array((-tan_vect[-1],tan_vect[0]))
norm_vect = norm_vect/np.linalg.norm(norm_vect,axis = 0)
coord_new = af.coordinates+(np.expand_dims(norm_vect,axis = -1)*offset).transpose(-1,0,1)

#%%

fig,ax = plt.subplots(1,1,figsize = (6.4,4.5))
ax.plot(af.coordinates[0],af.coordinates[-1])
# ax.scatter(af.coordinates[0,:30],af.coordinates[1,:30]-dx)
# ax.scatter(af.coordinates[0,:30],af.coordinates[1,:30]-2*dx)
# ax.scatter(af.coordinates[0,:30],af.coordinates[1,:30]-3*dx)

ax.scatter(af.coordinates[0],np.ones(len(af.coordinates[0]))*0)
ax.scatter(af.coordinates[0],np.ones(len(af.coordinates[0]))*dx)
ax.scatter(af.coordinates[0],np.ones(len(af.coordinates[0]))*-dx)
ax.scatter(af.coordinates[0],np.ones(len(af.coordinates[0]))*-2*dx)
ax.scatter(af.coordinates[0],np.ones(len(af.coordinates[0]))*2*dx)
ax.scatter(af.coordinates[0],np.ones(len(af.coordinates[0]))*-3*dx)
ax.scatter(af.coordinates[0],np.ones(len(af.coordinates[0]))*3*dx)


# # ax.scatter(af.coordinates[:30,0],af.coordinates[:30,-1]-3*dx)

# ax.scatter(af.coordinates[0,30:],af.coordinates[1,30:]+dx)
# ax.scatter(af.coordinates[0,30:],af.coordinates[1,30:]+dx)
# ax.scatter(af.coordinates[0,30:],af.coordinates[1,30:]+dx)

# ax.scatter(af.coordinates[30:,0],af.coordinates[30:,-1]+2*dx)
# ax.scatter(af.coordinates[30:,0],af.coordinates[30:,-1]+3*dx)

ax.scatter(coord_new[:,0],coord_new[:,-1])

ax.quiver(af.coordinates[0],af.coordinates[-1],-tan_vect[-1],tan_vect[0])
# ax.plot(blade_nodes[0,:,1],blade_nodes[0,:,2])
# ax.plot(blade_nodes_tw[0,:,1],blade_nodes_tw[0,:,2])
# ax.set_xlabel('y')
# ax.set_ylabel('z')
# ax.set_xlim([0,0.015])
# ax.set_ylim([-0.0015,0.0015])
plt.grid()

#%% Airfoil parameters

# chord length
c = 1
# thickness relative to chord length (t/c)
t = .12
# chamber 
h = 0*c

#%% Configure Joukowski airfoil 

# radius of circle in zeta-plane
a = c/4
# thickness parameter (offset of circle left of the y-axis in zeta-plane)
e = 4/(3*np.sqrt(3))*t
# magnitude of offset from the origin in the zeta-axis
m = np.sqrt((e*a)**2+(h/2)**2)
# angle to new center of circle in zeta_axis
delta = np.arccos(-e*a/m)
# scaled radius to ensure that it intersects the x-axis at x=a. This ensurea a sharp trailing edge. 
R = abs(1-m/a*np.e**(1j*delta))

# angle vector [rad]
th = np.arange(360)*np.pi/180
# transformed circle in the zeta-plane
zeta_af = R*a*np.e**(1j*th)+m*np.e**(1j*delta)
# Joukowski transform into z-plane
z_af = zeta_af+a**2/zeta_af

# transformed circle in the zeta-plane
zeta_af_2 = .95*R*a*np.e**(1j*th)
# Joukowski transform into z-plane
z_af_2 = zeta_af_2+a**2/zeta_af_2

fig,ax = plt.subplots(1,1, figsize = (6.5,4.5))
ax.plot(np.real(a*np.e**(1j*th)),np.imag(a*np.e**(1j*th)))
ax.plot(np.real(zeta_af),np.imag(zeta_af))
ax.plot(np.real(z_af),np.imag(z_af))

ax.plot(np.real(zeta_af_2),np.imag(zeta_af_2))
ax.plot(np.real(z_af_2),np.imag(z_af_2))

ax.scatter(0,0)
ax.scatter(np.real(m*np.e**(1j*delta)),np.imag(m*np.e**(1j*delta)))
# ax.axis([-1.2,1.2,-1.2,1.2])
ax.grid()
