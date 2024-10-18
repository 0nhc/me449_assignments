import numpy as np
import modern_robotics as mr


# Known values
w_1 = np.array([0, 0, 1])
w_2 = np.array([0, 1, 0])
w_3 = np.array([0, 1, 0])
w_4 = np.array([0, 1, 0])
w_5 = np.array([0, 0, -1])
w_6 = np.array([0, 1, 0])

R_13 = np.array([[-0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, -0.7071]])
R_s2 = np.array([[-0.6964, 0.1736, 0.6964], [-0.1228, -0.9848, 0.1228], [0.7071, 0, 0.7071]])
R_25 = np.array([[-0.7566, -0.1198, -0.6428], [-0.1564, 0.9877, 0], [0.6348, 0.1005, -0.7661]])
R_12 = np.array([[0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, 0.7071]])
R_34 = np.array([[0.6428, 0, -0.7660], [0, 1, 0], [0.7660, 0, 0.6428]])
R_s6 = np.array([[0.9418, 0.3249, -0.0859], [0.3249, -0.9456, -0.0151], [-0.0861, -0.0136, -0.9962]])
R_6b = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])


class Part2:
    """
    Class to calculate the forward kinematics of the robot    
    """
    def __init__(self):
        # Initialize the forward kinematics R matrices
        self.R = []
        self._init_R_with_known_values()
        
        # Initialize the w vectors
        self.w = []
        self._init_w_with_known_values()
        
    
    def _init_w_with_known_values(self):
        """
        Initialize the w vectors
        """
        self.w = [w_1, w_2, w_3, w_4, w_5, w_6]
        
        
    def _init_R_with_known_values(self):
        """
        Initialize the forward kinematics R matrices with the known values
        """
        # R_s1: R_s1*R_12 = R_s2
        R_s1 = R_s2 @ np.linalg.inv(R_12)

        # R_12 is already given

        # R_23: R_12*R_23 = R_13
        R_23 = np.linalg.inv(R_12) @ R_13

        # R_34 is already given

        # R_45: R_25 = R_23*R_34*R_45
        R_45 = np.linalg.inv(R_23 @ R_34) @ R_25

        # R_56: R_s6 = R_s1*R_12*R_23*R_34*R_45*R_56
        R_56 = np.linalg.inv(R_s1 @ R_12 @ R_23 @ R_34 @ R_45) @ R_s6

        # R_6b is already given
        
        # Store all the R matrices
        self.R = [R_s1, R_12, R_23, R_34, R_45, R_56]
    
    
    def get_R_sb(self):
        """
        Get the R_sb matrix
        """
        R_s6 = self.R[0]
        for R in self.R[1:]:
            R_s6 = R_s6 @ R
        R_sb = R_s6 @ R_6b
        return R_sb
    
    
    def R_to_theta(self):
        """
        Calculate the theta values from the R matrices
        """
        theta_list = []
        for idx, R in enumerate(self.R):
            # find theta
            omega_hat_skew_dot_th = mr.MatrixLog3(R)
            omega_dot_th = mr.so3ToVec(omega_hat_skew_dot_th)
            omega = self.w[idx]
            theta = np.linalg.norm(omega_dot_th)
            theta_sign = np.sign(np.dot(omega_dot_th, omega))
            theta = theta_sign * theta
            theta_list.append(theta)
        return theta_list


if __name__ == "__main__":
    # Set numpy printing options
    np.set_printoptions(precision=3, suppress=True)
    
    # Calculate the results
    pt2 = Part2()
    
    # Get the theta values
    theta = pt2.R_to_theta()
    print(f"Theta values: \n{theta}\n")
    
    # Get the R_sb matrix
    R_sb = pt2.get_R_sb()
    print(f"R_sb: \n{R_sb}\n")