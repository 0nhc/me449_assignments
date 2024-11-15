import numpy as np
import modern_robotics as mr


# Parameters from https://hades.mech.northwestern.edu/images/d/d9/UR5-parameters-py.txt
M01 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]
M12 = [[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]]
M23 = [[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]]
M34 = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]]
M45 = [[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]]
M56 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]]
M67 = [[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]]
G1 = np.diag([0.010267495893, 0.010267495893,  0.00666, 3.7, 3.7, 3.7])
G2 = np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393])
G3 = np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275])
G4 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G5 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G6 = np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])
Glist = [G1, G2, G3, G4, G5, G6]
Mlist = [M01, M12, M23, M34, M45, M56, M67] 
Slist = [[0,         0,         0,         0,        0,        0],
        [0,         1,         1,         1,        0,        1],
        [1,         0,         0,         0,       -1,        0],
        [0, -0.089159, -0.089159, -0.089159, -0.10915, 0.005491],
        [0,         0,         0,         0,  0.81725,        0],
        [0,         0,     0.425,   0.81725,        0,  0.81725]]


def Puppet(thetalist,
           dthetalist,
           g,
           Mlist,
           Slist,
           Glist,
           t,
           dt,
           damping,
           stiffness,
           restLength):
    """
    Simulate the system with the given parameters.

    Args:
        thetalist: A list of initial joint angles.
        dthetalist: A list of initial joint velocities.
        g: The gravity vector.
        Mlist: A list of link transposes.
        Slist: A list of joint screw axes.
        Glist: A list of link inertia matrices.
        t: The total time of the simulation.
        dt: The timestep of the simulation.
        damping: The damping coefficient.
        stiffness: The spring stiffness.
        restLength: The spring rest length.

    Returns:
        thetamat: A matrix of joint angles at each timestep.
        dthetamat: A matrix of joint velocities at each timestep.
    """
    assert len(thetalist) == len(dthetalist)
    n = len(thetalist)
    N = int(t/dt)

    thetamat = np.zeros((N + 1, n))
    thetamat[0] = thetalist
    
    dthetamat = np.zeros((N + 1, n))
    dthetamat[0] = dthetalist

    for idx in range(N):
    #    print(str(round(idx/N*100, 2))+' %')
        damping_t = -damping * dthetalist
        thdd = mr.ForwardDynamics(
            thetalist,
            dthetalist,
            damping_t,
            g,
            np.zeros(n),
            Mlist,
            Glist,
            Slist)
        
        th, dth = mr.EulerStep(
            thetalist,
            dthetalist,
            thdd,
            dt)
        
        thetalist = th
        dthetalist = dth
        
        thetamat[idx+1] = thetalist
        dthetamat[idx+1] = dthetalist

    return thetamat, dthetamat


def main():
    thetalist = np.array([0, 0, 0, 0, 0, 0])
    dhetalist = np.array([0, 0, 0, 0, 0, 0])
    g = np.array([0, 0, -9.81])

    thetamat, dthetamat = Puppet(
        thetalist,
        dhetalist,
        g,
        Mlist,
        Slist,
        Glist,
        5,
        0.001,
        0.0,
        None,
        None)
    
    np.savetxt('part1a.csv', thetamat, delimiter=',')

    thetamat, dthetamat = Puppet(
        thetalist,
        dhetalist,
        g,
        Mlist,
        Slist,
        Glist,
        5,
        0.05,
        0.0,
        None,
        None)
    
    np.savetxt('part1b.csv', thetamat, delimiter=',')


if __name__ == '__main__':
    main()
