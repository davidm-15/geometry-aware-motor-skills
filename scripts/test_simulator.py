from data_generation.simulation import VirtualEndEffector
import numpy as np
from matplotlib import pyplot as plt


# python -m scripts.test_simulator







if __name__ == "__main__":
    mass = 10  # kg
    I_matrix = [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]  # kg*m^2
    cv = 0.1  # N/(m/s)
    comega = 0.1  # Nm/(rad/s)
    dt = 0.01  # s
    lookahead_distance = 0.1  # m
    kp_linear = 1.0  # N/m
    kp_angular = 0.5  # Nm/rad


    simulator = VirtualEndEffector(
        m=mass, 
        I=I_matrix, 
        cv=cv, 
        comega=comega, 
        dt=dt, 
        lookahead_distance=lookahead_distance,
        kp_linear=kp_linear,
        kp_angular=kp_angular
    )

    steps = 100


    f_ctrl = np.zeros((steps, 3))  # N
    f_ctrl[:, 0] = np.sin(np.linspace(0, 2 * np.pi, steps))
    # f_ctrl[:, 0] = 10.0  # N constant force in x-direction

    tau_ctrl = np.array([0.0, 0.0, 0.5])  # Nm

    simulator.reset(start_pos=[0, 0, 0], start_quat=[0, 0, 0, 1])
    times = [0.0]
    positions = [simulator.p.copy()]
    velocities = [simulator.v.copy()]
    omegas = [simulator.omega.copy()]

    for step in range(steps):
        simulator.step(f_ctrl[step], tau_ctrl)
        times.append(simulator.time)
        positions.append(simulator.p.copy())
        velocities.append(simulator.v.copy())
        omegas.append(simulator.omega.copy())

    positions = np.array(positions)
    velocities = np.array(velocities)
    omegas = np.array(omegas)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(times, positions)
    plt.title("Position vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend(['x', 'y', 'z'])
    plt.grid()
    
    plt.subplot(3, 1, 2)
    plt.plot(times, velocities)
    plt.title("Velocity vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend(['vx', 'vy', 'vz'])
    plt.grid() 

    plt.subplot(3, 1, 3)
    plt.plot(times, velocities)
    plt.title("Velocity vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid()

    plt.tight_layout()
    plt.show()