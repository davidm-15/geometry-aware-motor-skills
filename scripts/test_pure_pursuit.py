from data_generation.pure_pursuit import PurePursuitController
import numpy as np
from matplotlib import pyplot as plt
from data_generation.simulation import VirtualEndEffector

# python -m scripts.test_pure_pursuit

if __name__ == "__main__":
    print("Testing Pure Pursuit Controller...")


    path = np.zeros((100, 7))


    path[:, 0] = np.linspace(0, 10, 100)
    path[:, 6] = 1.0


    simulator = VirtualEndEffector(m=1.0, 
                                   I=np.eye(3), 
                                   cv=5.0, 
                                   comega=2.0,
                                   dt=0.1, 
                                   lookahead_distance=1.0, 
                                   kp_linear=10.0, 
                                   kp_angular=10.0)

    current_pos = np.array([0.0, 0.0, 0.0])
    current_quat = np.array([0.0, 0.0, 0.0, 1.0])   

    simulator.reset(current_pos, current_quat)

    f_ctrls = []
    tau_ctrls = []
    velocities = [simulator.v.copy()]
    positions = [current_pos.copy()]
    accelerations = []

    current_idx = 0

    for t in range(200):
        target_pos, target_quat, idx = simulator.controller.get_lookahead_point(current_pos, path[:, 0:3], path[:, 3:8], start_idx=current_idx)
        current_idx = idx
        f_ctrl, tau_ctrl = simulator.controller.compute_control(current_pos, current_quat, target_pos, target_quat)

        simulator.step(f_ctrl, tau_ctrl)
        current_pos = simulator.p
        current_quat = simulator.q
        
        print(f"Time {t*0.1:.2f}s: Target Pos {target_pos}, Control Force {f_ctrl}, Control Torque {tau_ctrl}")


        f_ctrls.append(f_ctrl)
        tau_ctrls.append(tau_ctrl)
        velocities.append(simulator.v.copy())
        positions.append(simulator.p.copy())

        acceleration = velocities[-1] - velocities[-2] if len(velocities) > 1 else np.zeros(3)
        acceleration = np.linalg.norm(acceleration)
        full_vel = np.linalg.norm(simulator.v)

        print(f"Current Index: {current_idx}, Path Length: {len(path)}")
        if (acceleration < 0.1) and (current_idx >= len(path) - 1) and (full_vel < 0.1):
            print(f"acceleration: {acceleration} current_idx: {current_idx}")
            print("Reached end of path with low acceleration. Stopping simulation.")
            break
            



    f_ctrls = np.array(f_ctrls)
    tau_ctrls = np.array(tau_ctrls)
    velocities = np.array(velocities)
    positions = np.array(positions)



    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(positions[:, 0], positions[:, 1], label='End Effector Trajectory')
    plt.plot(path[:, 0], path[:, 1], label='Path', linestyle='--')
    plt.title('End Effector Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(f_ctrls)
    plt.title('Control Forces')
    plt.xlabel('Time Step')
    plt.ylabel('Force')

    plt.subplot(2, 2, 3)
    plt.plot(tau_ctrls)
    plt.title('Control Torques')
    plt.xlabel('Time Step')
    plt.ylabel('Torque')

    plt.subplot(2, 2, 4)
    plt.plot(velocities)
    plt.title('End Effector Velocities')
    plt.xlabel('Time Step')
    plt.ylabel('Velocity')

    plt.tight_layout()
    plt.show()