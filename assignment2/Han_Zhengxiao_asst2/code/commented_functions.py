import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt
import random

# Set print options for numpy arrays
np.set_printoptions(precision=3, suppress=True)


"""Constants:"""
L1 = 0.425
L2 = 0.392
H1 = 0.089
H2 = 0.095
W1 = 0.109
W2 = 0.082

# Home Configuration:
M = np.array([[-1, 0, 0, L1+L2],
              [0, 0, 1, W1+W2],
              [0, 1, 0, H1-H2],
              [0, 0, 0, 1]])

# B:
B_list = []
B_list.append(np.array([0, 1, 0, W1+W2, 0, L1+L2]))
B_list.append(np.array([0, 0, 1, H2, -L1-L2, 0]))
B_list.append(np.array([0, 0, 1, H2, -L2, 0]))
B_list.append(np.array([0, 0, 1, H2, 0, 0]))
B_list.append(np.array([0, -1, 0, -W2, 0, 0]))
B_list.append(np.array([0, 0, 1, 0, 0, 0]))
B = np.column_stack(B_list)

# T_sd:
T_sd = np.array([[1, 0, 0, 0.3],
                 [0, 1, 0, 0.3],
                 [0, 0, 1, 0.4],
                 [0, 0, 0, 1]])


def pi2pi(angle):
    """
    Maps an angle to the range -pi to pi.

    Args:
        angle: The angle in radians.
    
    Returns:
        The angle in radians, mapped to the range -pi to pi.
    """
    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle

def print_iteration(M, B, V_b, index, joint_vector):
    """
    Prints the current iteration of the inverse kinematics algorithm.

    Args:
        M: The home configuration of the end-effector.
        B: The screw axes in the end-effector frame.
        V_b: The error twist.
        index: The current iteration index.
        joint_vector: The current joint vector.
    
    Returns:
        None
    """
    joint_vector = np.array(joint_vector)
    joint_vector = joint_vector.tolist()

    print(f"Iteration {index}:\n")

    print("joint vector:")
    joint_vector_to_print = joint_vector
    for i in range(len(joint_vector_to_print)):
        joint_vector_to_print[i] = round(joint_vector_to_print[i], 3)
    print(f"{str(joint_vector_to_print)[1:-1]}\n")

    print("SE(3) end-effector config:")
    Tsb = mr.FKinBody(M, B, joint_vector)
    decimal_places = 3
    width = 6
    print('\n'.join(' '.join(f"{cell:{width}.{decimal_places}f}" for cell in row) for row in Tsb))
    print()

    decimal_places = 3
    Vb_to_print = V_b.tolist()
    for i in range(len(Vb_to_print)):
        Vb_to_print[i] = round(Vb_to_print[i], decimal_places)
    Vb_to_print = tuple(Vb_to_print)
    print(f"          error twist V_b: {Vb_to_print}")
    print(f"angular error ||omega_b||: {np.linalg.norm([V_b[0], V_b[1], V_b[2]])}")
    print(f"     linear error ||v_b||: {np.linalg.norm([V_b[3], V_b[4], V_b[5]])}\n")
    print("=========================================================================")

def save_csv(filename, data):
    """
    Saves a numpy array to a CSV file.

    Args:
        filename: The name of the file to save the data to.
        data: The numpy array to save.

    Returns:
        None
    """
    np.savetxt(filename, data, delimiter=',')

def Vb(M, B, T_sd, joint_vector):
    """
    Calculates the error twist V_b.

    Args:
        M: The home configuration of the end-effector.
        B: The screw axes in the end-effector frame.
        T_sd: The desired end-effector configuration.
        joint_vector: The current joint vector.

    Returns:
        The error twist V_b.
    """
    Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, B, joint_vector)), T_sd)))
    return Vb

def check_convergence(V_b, eps_w, eps_v):
    """
    Checks if the error twist V_b is below the convergence thresholds.

    Args:
        V_b: The error twist.
        eps_w: The angular convergence threshold.
        eps_v: The linear convergence threshold.

    Returns:
        True if the error twist is above the convergence thresholds, False otherwise.
    """
    return (np.linalg.norm([V_b[0], V_b[1], V_b[2]]) > eps_w \
         or np.linalg.norm([V_b[3], V_b[4], V_b[5]]) > eps_v)

def IKinBodyIterates(M, B, T_sd, theta_0, traj_filename, max_iterations=100, eps_w=1e-3, eps_v=1e-4, printing='true'):
    """
    Calculates the joint vector that achieves the desired end-effector configuration.

    Args:
        M: The home configuration of the end-effector.
        B: The screw axes in the end-effector frame.
        T_sd: The desired end-effector configuration.
        theta_0: The initial guess of the joint vector.
        traj_filename: The filename to save the trajectory to.
        max_iterations: The maximum number of iterations.
        eps_w: The angular convergence threshold.
        eps_v: The linear convergence threshold.
        printing: Whether to print the iteration information.

    Returns:
        The number of iterations it took to converge.
    """
    # Initialize variables
    index = 1
    joint_vector = np.array(theta_0)
    joint_vector = joint_vector.tolist()
    V_b = Vb(M, B, T_sd, theta_0)
    iterating = check_convergence(V_b, eps_w, eps_v)

    # For saving later
    traj = np.array(joint_vector)
    angular_error = np.array(np.linalg.norm([V_b[0], V_b[1], V_b[2]]))
    linear_error = np.array(np.linalg.norm([V_b[3], V_b[4], V_b[5]]))

    # Iterate until convergence
    while iterating and index < max_iterations:
        # Update joint vector
        joint_vector = joint_vector + np.dot(np.linalg.pinv(mr.JacobianBody(B, joint_vector)), V_b)
        joint_vector = [np.arctan2(np.sin(theta), np.cos(theta)) for theta in joint_vector]

        # Update error twist
        V_b = Vb(M, B, T_sd, joint_vector)

        # Print iteration information
        if(printing):
            print_iteration(M, B, V_b, index, joint_vector)
        
        # Save data
        traj = np.vstack([traj, joint_vector])
        angular_error = np.vstack([angular_error, np.array(np.linalg.norm([V_b[0], V_b[1], V_b[2]]))])
        linear_error = np.vstack([linear_error, np.array(np.linalg.norm([V_b[3], V_b[4], V_b[5]]))])
        iterating = check_convergence(V_b, eps_w, eps_v)

        # Check if converged
        if(iterating == False):
            if(printing):
                print(f"Converged. It took {index} iterations.")
                print("Trajectory:")
                print(traj)
                save_csv(traj_filename+'.csv', traj)
                save_csv(traj_filename+'_angular_error.csv', angular_error)
                save_csv(traj_filename+'_linear_error.csv', linear_error)
            break
        
        index += 1
    
    return index

def joint_traj2position_traj(joint_traj):
    """
    Converts a joint trajectory to a position trajectory.

    Args:
        joint_traj: The joint trajectory.
    
    Returns:
        The position trajectory.
    """
    joint_vector = joint_traj[0]
    pose_matrix = mr.FKinBody(M, B, joint_vector)
    position = pose_matrix[:3, 3]
    position_traj = np.array(position)
    for idx in range(1, joint_traj.shape[0]):
        joint_vector = joint_traj[idx]
        pose_matrix = mr.FKinBody(M, B, joint_vector)
        position = pose_matrix[:3, 3]
        position_traj = np.vstack([position_traj, position])
    return position_traj

def plot_position_traj(long_filename, short_filename):
    """
    Plots two position trajectories in 3D space.

    Args:
        long_filename: The filename of the long trajectory.
        short_filename: The filename of the short trajectory.

    Returns:
        None
    """
    long_angle_traj = np.genfromtxt(long_filename+'.csv', delimiter=',')
    short_angle_traj = np.genfromtxt(short_filename+'.csv', delimiter=',')
    long_position_traj = joint_traj2position_traj(long_angle_traj)
    short_position_traj = joint_traj2position_traj(short_angle_traj)

    # Plot two trajectories in 3D space
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(long_position_traj[:, 0], long_position_traj[:, 1], long_position_traj[:, 2], label='long_traj')
    ax.plot(short_position_traj[:, 0], short_position_traj[:, 1], short_position_traj[:, 2], label='short_traj')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def plot_angular_error(long_filename, short_filename):
    """
    Plots the angular error of two trajectories.

    Args:
        long_filename: The filename of the long trajectory.
        short_filename: The filename of the short trajectory.

    Returns:
        None
    """
    long_angular_error = np.genfromtxt(long_filename+'_angular_error.csv', delimiter=',')
    short_angular_error = np.genfromtxt(short_filename+'_angular_error.csv', delimiter=',')
    plt.plot(long_angular_error, label='long_traj')
    plt.plot(short_angular_error, label='short_traj')
    plt.xlabel('Iterations')
    plt.ylabel('Angular Error')
    plt.legend()
    plt.show()

def plot_linear_error(long_filename, short_filename):
    """
    Plots the linear error of two trajectories.

    Args:
        long_filename: The filename of the long trajectory.
        short_filename: The filename of the short trajectory.

    Returns:
        None
    """
    long_linear_error = np.genfromtxt(long_filename+'_linear_error.csv', delimiter=',')
    short_linear_error = np.genfromtxt(short_filename+'_linear_error.csv', delimiter=',')
    plt.plot(long_linear_error, label='long_traj')
    plt.plot(short_linear_error, label='short_traj')
    plt.xlabel('Iterations')
    plt.ylabel('Linear Error')
    plt.legend()
    plt.show()

def filter_initial_guess(lb, ub):
    """
    Filters the initial guess to be within the bounds.

    Args:
        lb: The lower bound.
        ub: The upper bound.

    Returns:
        A random initial guess within the bounds.
    """
    # give a random initial guess
    ites = 0
    while (ites>=ub or ites<=lb):
        random_initial_guess = [random.uniform(-np.pi, np.pi) for i in range(6)]
        # round to 3 decimal places
        random_initial_guess = [round(i, 3) for i in random_initial_guess]
        random_traj_name = "random_traj"
        ites = IKinBodyIterates(M, B, T_sd, random_initial_guess, random_traj_name, printing=False)
    print(f"Found initial guess: {random_initial_guess} with {ites} iterations to converge.")
    return random_initial_guess

