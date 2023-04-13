
#!/usr/bin/env python

import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
import math, time, random
from scipy.spatial import Delaunay

def check_obst_is_in_FOV(agent_pos, agent_quat, obst_pos, fov_x_deg, fov_y_deg, fov_depth):
    """
    Check if the obstacle is in the agent's FOV.
    """
    # Compute the FOV vertices (from panther_ros.cpp)
    delta_y = fov_depth * abs(math.tan((fov_x_deg * math.pi / 180) / 2.0))
    delta_z = fov_depth * abs(math.tan((fov_y_deg * math.pi / 180) / 2.0))

    v0 = agent_pos
    v1 = np.array([-delta_y, delta_z, fov_depth])
    v2 = np.array([delta_y, delta_z, fov_depth])
    v3 = np.array([delta_y, -delta_z, fov_depth])
    v4 = np.array([-delta_y, -delta_z, fov_depth])

    # Rotate the FOV vertices
    v1 = rotate_vector(v1, agent_quat)
    v2 = rotate_vector(v2, agent_quat)
    v3 = rotate_vector(v3, agent_quat)
    v4 = rotate_vector(v4, agent_quat)

    # Check if the obstacle is in the FOV
    poly = np.array([v0, v1, v2, v3, v4])
    point = obst_pos
    return Delaunay(poly).find_simplex(point) >= 0  # True if point lies within poly

def rotate_vector(v, q):
    """
    Rotate a vector v by a quaternion q.
    """
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
    v_prime = quaternion_multiply(quaternion_multiply(q, np.append(v, 0)), q_conj)
    return v_prime[:3]

def quaternion_multiply(q0, q1):
    """
    Multiply two quaternions.
    """
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1

    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    return np.array([Q0Q1_x, Q0Q1_y, Q0Q1_z, Q0Q1_w])

def visualization(agent_pos, agent_quat, obst_pos, fov_x_deg, fov_y_deg, fov_depth):
    # visualize the agent's FOV
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('FOV visualization')
    ax.set_box_aspect([1,1,1])
    ax.scatter(agent_pos[0], agent_pos[1], agent_pos[2], c='r', marker='o')
    ax.scatter(obst_pos[0], obst_pos[1], obst_pos[2], c='b', marker='o')
    ax.plot([agent_pos[0], obst_pos[0]], [agent_pos[1], obst_pos[1]], [agent_pos[2], obst_pos[2]], c='b', linestyle='--')
    
    # Compute the FOV vertices (from panther_ros.cpp)

    delta_y = fov_depth * abs(math.tan((fov_x_deg * math.pi / 180) / 2.0))
    delta_z = fov_depth * abs(math.tan((fov_y_deg * math.pi / 180) / 2.0))

    v0 = agent_pos
    v1 = np.array([-delta_y, delta_z, fov_depth])
    v2 = np.array([delta_y, delta_z, fov_depth])
    v3 = np.array([delta_y, -delta_z, fov_depth])
    v4 = np.array([-delta_y, -delta_z, fov_depth])

    # Rotate the FOV vertices
    v1 = rotate_vector(v1, agent_quat)
    v2 = rotate_vector(v2, agent_quat)
    v3 = rotate_vector(v3, agent_quat)
    v4 = rotate_vector(v4, agent_quat)

    ax.scatter(v0[0], v0[1], v0[2], c='g', marker='o')
    ax.scatter(v1[0], v1[1], v1[2], c='g', marker='o')
    ax.scatter(v2[0], v2[1], v2[2], c='g', marker='o')
    ax.scatter(v3[0], v3[1], v3[2], c='g', marker='o')
    ax.scatter(v4[0], v4[1], v4[2], c='g', marker='o')
    ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], c='g', linestyle='-')
    ax.plot([v0[0], v2[0]], [v0[1], v2[1]], [v0[2], v2[2]], c='g', linestyle='-')
    ax.plot([v0[0], v3[0]], [v0[1], v3[1]], [v0[2], v3[2]], c='g', linestyle='-')
    ax.plot([v0[0], v4[0]], [v0[1], v4[1]], [v0[2], v4[2]], c='g', linestyle='-')
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], c='g', linestyle='-')
    ax.plot([v2[0], v3[0]], [v2[1], v3[1]], [v2[2], v3[2]], c='g', linestyle='-')
    ax.plot([v3[0], v4[0]], [v3[1], v4[1]], [v3[2], v4[2]], c='g', linestyle='-')
    ax.plot([v4[0], v1[0]], [v4[1], v1[1]], [v4[2], v1[2]], c='g', linestyle='-')
    ax.view_init(30, 30)
    plt.show()

if __name__ == "__main__":

    ##
    ## Define FOV-related parameters
    ##

    fov_x_deg = 76.0
    fov_y_deg = 47.0
    fov_depth = 5.0

    ##
    ## unit test
    ##

    # for i in range(100):
    #     agent_pos = np.array([0.0, 0.0, 0.0])
    #     agent_quat = quaternion_from_euler(random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi))
    #     obst_pos = np.array([random.uniform(-fov_depth, fov_depth), random.uniform(-fov_depth, fov_depth), random.uniform(-fov_depth, fov_depth)])
    #     print(check_obst_is_in_FOV(agent_pos, agent_quat, obst_pos, fov_x_deg, fov_y_deg, fov_depth))
    #     visualization(agent_pos, agent_quat, obst_pos, fov_x_deg, fov_y_deg, fov_depth)
    #     time.sleep(0.1)


    # Testing points on a sphere
    agent_pos = np.array([5.0, -4.0, 3.0])
    agent_quat = quaternion_from_euler(random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi))

    # Sample obstacle positions on the surface of a sphere around the agent position
    sphere_radius = 6.0
    sphere_center = agent_pos

    obst_pos = []
    resolution = 50
    for phi in range(resolution):
        for theta in range(resolution):
            obst_pos.append(np.array(
                    [
                        sphere_center[0] + sphere_radius * math.sin(2 * math.pi * phi / resolution) * math.cos(2 * math.pi * theta / resolution),
                        sphere_center[1] + sphere_radius * math.sin(2 * math.pi * phi / resolution) * math.sin(2 * math.pi * theta / resolution),
                        sphere_center[2] + sphere_radius * math.cos(2 * math.pi * phi / resolution)
                    ]
                )
            )

    in_FOV_points = []
    for pos in obst_pos:
        if check_obst_is_in_FOV(agent_pos, agent_quat, pos, fov_x_deg, fov_y_deg, fov_depth):
            in_FOV_points.append(pos)
    
    visualization(agent_pos, agent_quat, in_FOV_points, fov_x_deg, fov_y_deg, fov_depth)

    



