import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axis3d import Axis
import numpy as np
from scipy.linalg import expm

# Patch to 3D axis to remove the margins around the x, y and z limits
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new

def rotate_by_euler(points, xyz):
    # Rotate vector v (or array of vectors) by the Euler angles xyz
    for theta, axis in zip(xyz, np.eye(3)):
        points = np.dot(np.array(points), expm(np.cross(np.eye(3), axis*-theta)).T)
    return points

def quad_3d_plot(results, fancy=True):
    # Set up axes grids
    fig = plt.figure(figsize=(20,15))
    ax = plt.subplot2grid((20, 40), (0, 0), colspan=24, rowspan=20, projection='3d')
    # ax2 = plt.subplot2grid((20, 40), (1, 28), colspan=12, rowspan=4)
    # ax3 = plt.subplot2grid((20, 40), (6, 28), colspan=12, rowspan=4)
    # ax4 = plt.subplot2grid((20, 40), (11, 28), colspan=12, rowspan=4)
    # ax5 = plt.subplot2grid((20, 40), (15, 28), colspan=12, rowspan=4)

    # Plot the trajectory of the quadcopter in the x, y, and z dimensions across the simulations
    c = 0.0
    plt.rcParams['grid.color'] = [c, c, c, 0.075]
    mpl.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.xmargin'] = 0
    plot_limit_xy = 14
    plot_limit_z = 30
    quad_size = 0.5
    n_points_rotor = 15
    points_quad_initial = [[-quad_size, -quad_size, 0], [-quad_size, quad_size, 0], 
                           [quad_size, quad_size, 0], [quad_size, -quad_size, 0]]
    points_rotor_initial = np.vstack((np.sin(np.linspace(0., 2.*np.pi, n_points_rotor)),
                                      np.cos(np.linspace(0., 2.*np.pi, n_points_rotor)),
                                      np.repeat(0.0, n_points_rotor))).T * quad_size * 0.8

    # Create 3D plot
    ax.view_init(12, -55)
    ax.dist = 7.6
    
    # Plot the position of the quadcopter
    x_limited = [x for x in results['x'] if np.abs(x) <= plot_limit_xy]
    y_limited = [y for y in results['y'] if np.abs(y) <= plot_limit_xy]
    z_limited = [z for z in results['z'] if z <= plot_limit_z]
    l = min(len(x_limited), len(y_limited))
    ax.plot(x_limited[0:l], y_limited[0:l], np.repeat(0.0, l), c='darkgray', linewidth=0.9)
    l = min(len(x_limited), len(z_limited))
    ax.plot(x_limited[0:l], np.repeat(plot_limit_xy, l), z_limited[0:l], c='darkgray', linewidth=0.9)
    l = min(len(y_limited), len(z_limited))
    ax.plot(np.repeat(-plot_limit_xy, l), y_limited[0:l], z_limited[0:l], c='darkgray', linewidth=0.9)
    
    # Plot 3D trajectory
    ax.plot(results['x'], results['y'], results['z'], c='gray', linewidth=0.5)
    n_timesteps = len(results['x'])
    colors = np.array([ [230, 25, 75, 255],
                        [60, 180, 75, 255],
                        [255, 225, 25, 255],
                        [0, 130, 200, 255]]) / 255.
    
    # Plot the quadcopter position as a dot on a trajectory for each full second
    for t in range(n_timesteps):
        if results['time'][t]%1.0 <= 0.025 or results['time'][t]%1.0 >= 0.975:
            ax.scatter([results['x'][t]], [results['y'][t]], [results['z'][t]], s=5, c=[0., 0., 0., 0.3])
        alpha1 = 0.96*np.power(t/n_timesteps, 20)+0.04
        alpha2 = 0.5 * alpha1
        
        # Plot the frame
        if fancy or t == n_timesteps -1:
            # Rotate the frame
            points_quad = rotate_by_euler(points_quad_initial, np.array([results['phi'][t], results['theta'][t], results['psi'][t]]))
            # Move the frame
            points_quad += np.array([results['x'][t], results['y'][t], results['z'][t]])
        
        # Plot the frame projections for last timestep
        if t == n_timesteps -1:
            # Z plane
            if np.abs(results['x'][t]) <= plot_limit_xy and np.abs(results['y'][t]) <= plot_limit_xy:
                ax.plot(points_quad[[0,2], 0], points_quad[[0,2], 1], [0., 0.], c=[0., 0., 0., 0.1])
                ax.plot(points_quad[[1,3], 0], points_quad[[1,3], 1], [0., 0.], c=[0., 0., 0., 0.1])
            # Y plane
            if np.abs(results['x'][t]) <= plot_limit_xy and np.abs(results['z'][t]) <= plot_limit_z:
                ax.plot(points_quad[[0,2], 0], [plot_limit_xy, plot_limit_xy], 
                         points_quad[[0,2], 2], c=[0., 0., 0., 0.1])
                ax.plot(points_quad[[1,3], 0], [plot_limit_xy, plot_limit_xy], 
                        points_quad[[1,3], 2], c=[0., 0., 0., 0.1])
            # X plane
            if np.abs(results['y'][t]) <= plot_limit_xy and np.abs(results['z'][t]) <= plot_limit_z:
                ax.plot([-plot_limit_xy, -plot_limit_xy], points_quad[[0,2], 1], 
                        points_quad[[0,2], 2], c=[0., 0., 0., 0.1])
                ax.plot([-plot_limit_xy, -plot_limit_xy], points_quad[[1,3], 1], 
                        points_quad[[1,3], 2], c=[0., 0., 0., 0.1])
        
        # Plot the frame for all other timesteps
        if fancy:
            ax.plot(points_quad[[0,2], 0], points_quad[[0,2], 1], 
                    points_quad[[0,2], 2], c=[0., 0., 0., alpha2])
            ax.plot(points_quad[[1,3], 0], points_quad[[1,3], 1], 
                    points_quad[[1,3], 2], c=[0., 0., 0., alpha2])
        
        # Plot rotors
        # Rotate rotor
        if fancy or t == n_timesteps -1:
            points_rotor = rotate_by_euler(points_rotor_initial, np.array([results['phi'][t], results['theta'][t], results['psi'][t]]))
        
        # Move rotor for each frame point
        for i, color in zip(range(4), colors):
            if fancy or t == n_timesteps -1:
                points_rotor_moved = points_rotor + points_quad[i]
            
            # Plot rotor projections
            if t == n_timesteps -1:
                # Z plane
                if np.abs(results['x'][t]) <= plot_limit_xy and np.abs(results['y'][t]) <= plot_limit_xy:
                    ax.add_collection3d(Poly3DCollection([list(zip(points_rotor_moved[:,0], points_rotor_moved[:,1], np.repeat(0, n_points_rotor)))], facecolor=[0.0, 0.0, 0.0, 0.1]))
                # Y plane
                if np.abs(results['x'][t]) <= plot_limit_xy and np.abs(results['z'][t]) <= plot_limit_z:
                    ax.add_collection3d(Poly3DCollection([list(zip(points_rotor_moved[:,0], np.repeat(plot_limit_xy, n_points_rotor), points_rotor_moved[:,2]))], facecolor=[0.0, 0.0, 0.0, 0.1]))
                # X plane
                if np.abs(results['y'][t]) <= plot_limit_xy and np.abs(results['z'][t]) <= plot_limit_z:
                    ax.add_collection3d(Poly3DCollection([list(zip(np.repeat(-plot_limit_xy, n_points_rotor), points_rotor_moved[:,1], points_rotor_moved[:,2]))], facecolor=[0.0, 0.0, 0.0, 0.1]))
            
            # Outline
            if t == n_timesteps-1:
                ax.plot(points_rotor_moved[:,0], points_rotor_moved[:,1], 
                        points_rotor_moved[:,2], c=color[0:3].tolist()+[alpha1], 
                        label='Rotor {:g}'.format(i+1))
            elif fancy:
                ax.plot(points_rotor_moved[:,0], points_rotor_moved[:,1], 
                        points_rotor_moved[:,2], c=color[0:3].tolist()+[alpha1])
            # Fill
            if fancy or t == n_timesteps -1:
                ax.add_collection3d(Poly3DCollection([list(zip(points_rotor_moved[:,0],
                                                               points_rotor_moved[:,1], 
                                                               points_rotor_moved[:,2]))],
                                                     facecolor=color[0:3].tolist()+[alpha2]))

    ax.legend()
    # ax.legend(bbox_to_anchor=(0.0 ,0.0 , 0.95, 0.85), loc='upper right')
    c = 'r'
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-plot_limit_xy, plot_limit_xy)
    ax.set_ylim(-plot_limit_xy, plot_limit_xy)
    ax.set_zlim(0, plot_limit_z)
    ax.set_xticks(np.arange(-plot_limit_xy, plot_limit_xy+2, 2))
    ax.set_yticks(np.arange(-plot_limit_xy, plot_limit_xy+2, 2))
    ax.set_zticks(np.arange(0, plot_limit_z+2, 2))
    
    # Plot the velocity of the quadcopter across the simulations
    # ax2.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    # ax2.plot(results['time'], results['x_velocity'], label='x_hat')
    # ax2.plot(results['time'], results['y_velocity'], label='y_hat')
    # ax2.plot(results['time'], results['z_velocity'], label='z_hat')
    # ax2.set_ylim(-20, 20)
    # ax2.legend()
    
    # Plot the rotation of the quadcopter over the x, y, and z axes across the simulations
    # ax3.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    # ax3.plot(results['time'], [a if a<= np.pi else a-2*np.pi for a in results['phi']], label='phi')
    # ax3.plot(results['time'], [a if a<= np.pi else a-2*np.pi for a in results['theta']], label='theta')
    # ax3.plot(results['time'], [a if a<= np.pi else a-2*np.pi for a in results['psi']], label='psi')
    # ax3.set_ylim(-np.pi, np.pi)
    # ax3.legend()
    
    # Plot the velocities (in radians per second) corresponding to each of the Euler angles
    # ax4.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    # ax4.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    # ax4.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    # ax4.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    # ax4.set_ylim(-3, 3)
    # ax4.legend()
    
    # Plot the agent's choice of rotor angles across the simulations
    # ax5.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
    # ax5.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
    # ax5.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
    # ax5.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
    # ax5.set_ylim(0, 1000)
    # ax5.legend()
    
    # Display!
    plt.show()
    