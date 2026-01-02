import numpy as np
import pandas as pd
import math
import tyro
import mujoco
from mujoco import viewer
from scipy.spatial.transform import Rotation as R
from drone_simulator import DroneSimulator
from pid import PID


SIM_TIME = 5000  # Maximum simulation time in steps


def xquat_to_euler(xquat):
    return R.from_quat([xquat[1], xquat[2], xquat[3], xquat[0]]).as_euler('xyz', degrees=True)


def build_world(fixed_track: bool, rotated_gates: bool) -> str:
    world = open("scene.xml").read()
    if not fixed_track:
        world = world.replace(
            '<body name="red_gate" pos="-2 0 1">',
            f'<body name="red_gate" pos="-2 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(0.7, 1.3)}">'
        )
        world = world.replace(
            '<body name="green_gate" pos="-4 -0.6 1.3">',
            f'<body name="green_gate" pos="-4 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(0.7, 1.3)}">'
        )
        world = world.replace(
            '<body name="blue_gate" pos="-6 0.6 0.7">',
            f'<body name="blue_gate" pos="-6 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(0.7, 1.3)}">'
        )

    if rotated_gates:
        world = world.replace(
            '<body name="red_gate"',
            f'<body name="red_gate" euler="0 0 {np.random.uniform(-45, 45) if not fixed_track else -15}"'
        )
        world = world.replace(
            '<body name="green_gate"',
            f'<body name="green_gate" euler="0 0 {np.random.uniform(-45, 45) if not fixed_track else -30}"'
        )
        world = world.replace(
            '<body name="blue_gate"',
            f'<body name="blue_gate" euler="0 0 {np.random.uniform(-45, 45) if not fixed_track else 45}"'
        )
    return world



def run_single_task(*, wind: bool, rotated_gates: bool, rendering_freq: float, fixed_track: bool) -> None:
    world = build_world(fixed_track, rotated_gates)
    model = mujoco.MjModel.from_xml_string(world)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    view = viewer.launch_passive(model, data)
    view.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    view.cam.fixedcamid = model.camera("track").id

    pos_targets = [
        [0, 0, 1], 
        data.body("red_gate").xpos.copy().tolist(),
        data.body("green_gate").xpos.copy().tolist(),
        data.body("blue_gate").xpos.copy().tolist(),
        [-8, 0, 1]
    ]

    yaw_quat_targets = [
        [1, 0, 0, 0],
        data.body("red_gate").xquat.copy().tolist(),
        data.body("green_gate").xquat.copy().tolist(),
        data.body("blue_gate").xquat.copy().tolist(),
        [1, 0, 0, 0]
    ]

    yaw_angle_targets = [xquat_to_euler(xquat)[2] for xquat in yaw_quat_targets]
    
    # TODO: Design PID control
    pid_roll = PID(
        gain_prop=5.0, gain_int=0.5, gain_der=1.0,
        sensor_period=model.opt.timestep, output_limits=(-1.0, 1.0)
    )

    pid_pitch = PID(
        gain_prop=5.0, gain_int=0.5, gain_der=0.3,
        sensor_period=model.opt.timestep, output_limits=(-1.0, 1.0)
    )

    pid_yaw = PID(
        gain_prop=5.0, gain_int=0.0, gain_der=0.5,
        sensor_period=model.opt.timestep, output_limits=(-1.0, 1.0)
    )
    
    pid_pos_x = PID(
        gain_prop=6.0, gain_int=0.2, gain_der=15.0,
        sensor_period=model.opt.timestep, output_limits=(-12.0, 12.0)
    )

    pid_pos_y = PID(
        gain_prop=4.0, gain_int=0.2, gain_der=10.0,
        sensor_period=model.opt.timestep, output_limits=(-3.0, 3.0)
    )

    pid_pos_z = PID(
        gain_prop=6.0, gain_int=3.0, gain_der=3.0,
        sensor_period=model.opt.timestep, output_limits=(-3.0, 3.0)
    )
    # END OF TODO

    # If you want the simulation to be displayed more slowly, decrease rendering_freq
    # Note that this DOES NOT change the timestep used to approximate the physics of the simulation!
    task_label = f"rotated={'yes' if rotated_gates else 'no'}, wind={'yes' if wind else 'no'}"
    print(f"Starting task ({task_label})")
    data.qpos[0:3] = pos_targets[0]
    data.qpos[3:7] = [1, 0, 0, 0]  # no rotation
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    wind_change_prob = 0.1 if wind else 0

    drone_simulator = DroneSimulator(
        model, data, view, wind_change_prob = wind_change_prob, rendering_freq = rendering_freq
    )

    # TODO: Define additional variables if needed
    current_target_idx = 0
    # END OF TODO

    try:
        for _ in range(SIM_TIME):
            current_pos, previous_pos = drone_simulator.position_sensor()
            current_orien, previous_orien = drone_simulator.orientation_sensor()
            
            if np.linalg.norm(np.array(current_pos) - np.array(pos_targets[-1])) < 0.2:
                break
            
            # TODO: define the current target position
            real_target_pos = np.array(pos_targets[current_target_idx])
            real_target_yaw = yaw_angle_targets[current_target_idx]
            
            # I tried midpoint aiming because initially, aiming directly at the gate wasn't working.
            # However, after I tuned the final PID parameters, it turned out to be unnecessary.
            # I'm keeping this code commented out just in case it's useful for the next assignment.
            #target_yaw_rad = np.radians(real_target_yaw)
            #target_normal = np.array([np.cos(target_yaw_rad), np.sin(target_yaw_rad), 0])
            
            #vec_target_to_drone = np.array(current_pos) - real_target_pos
            #dist_along_normal = np.dot(vec_target_to_drone, target_normal)
            
            #closest_point_on_line = real_target_pos + dist_along_normal * target_normal
            #closest_point_on_line[2] = real_target_pos[2]
            
            #dist_to_line = np.linalg.norm(np.array(current_pos) - closest_point_on_line)     
            #if dist_to_line > 0.3:
            #    effective_target_pos = (closest_point_on_line + real_target_pos) / 2.0
            #else:
            #    effective_target_pos = real_target_pos

            effective_target_pos = real_target_pos

            dx = effective_target_pos[0] - current_pos[0]
            dy = effective_target_pos[1] - current_pos[1]
            effective_target_yaw = np.degrees(np.arctan2(dy, dx)) + 180.0

            dist_to_real_target = np.linalg.norm(np.array(current_pos) - real_target_pos)
            if dist_to_real_target < 0.2 and current_target_idx < len(pos_targets) - 1:
                current_target_idx += 1
            # END OF TODO

            # TODO: use PID controllers to steer the drone
            thrust_correction = pid_pos_z.output_signal(effective_target_pos[2], [current_pos[2], previous_pos[2]])
            desired_thrust = 3.2496 + thrust_correction

            # Yaw
            current_yaw_deg = current_orien[2]
            previous_yaw_deg = previous_orien[2]

            yaw_error = (effective_target_yaw - current_yaw_deg + 180) % 360 - 180
            prev_yaw_error = (effective_target_yaw - previous_yaw_deg + 180) % 360 - 180
            
            yaw_thrust = pid_yaw.output_signal(0, [-yaw_error, -prev_yaw_error])

            # Position Error
            err_x_global_curr = effective_target_pos[0] - current_pos[0]
            err_y_global_curr = effective_target_pos[1] - current_pos[1]
            err_x_global_prev = effective_target_pos[0] - previous_pos[0]
            err_y_global_prev = effective_target_pos[1] - previous_pos[1]

            # Coordinate Transformation
            yaw_rad_curr = np.radians(current_yaw_deg)

            # Transform global position errors into the drone's local body frame.
            err_x_body_curr = err_x_global_curr * np.cos(yaw_rad_curr) + err_y_global_curr * np.sin(yaw_rad_curr)
            err_y_body_curr = -err_x_global_curr * np.sin(yaw_rad_curr) + err_y_global_curr * np.cos(yaw_rad_curr)

            err_x_body_prev = err_x_global_prev * np.cos(yaw_rad_curr) + err_y_global_prev * np.sin(yaw_rad_curr)
            err_y_body_prev = -err_x_global_prev * np.sin(yaw_rad_curr) + err_y_global_prev * np.cos(yaw_rad_curr)

            desired_pitch = pid_pos_x.output_signal(0, [-err_x_body_curr, -err_x_body_prev])            
            desired_roll = -pid_pos_y.output_signal(0, [-err_y_body_curr, -err_y_body_prev])

            # We want to:
            # - don't fly forward, if we aren't facing the target,
            # - slow down when getting close to the target to avoid overshooting.
            pitch_factor = 1.0
            if abs(yaw_error) > 45.0:
                pitch_factor = 0.0
            elif abs(yaw_error) > 20.0:
                pitch_factor = 0.4
            
            dist_to_effective = np.linalg.norm(np.array(current_pos) - effective_target_pos)
            if dist_to_effective < 1.0:
                brake_factor = 0.2 + 0.8 * dist_to_effective
                pitch_factor = min(pitch_factor, brake_factor)

            desired_pitch = desired_pitch * pitch_factor

            roll_thrust = -pid_roll.output_signal(desired_roll, [current_orien[0], previous_orien[0]])
            pitch_thrust = -pid_pitch.output_signal(desired_pitch, [current_orien[1], previous_orien[1]])
            # END OF TODO
            
            # For debugging purposes you can uncomment, but keep in mind that this slows down the simulation
            
            # data = np.array([pos_target + [desired_roll, desired_pitch, desired_yaw], np.concat([current_pos, current_orien])]).T
            # row_names = ["x", "y", "z", "roll", "pitch", "yaw"]
            # headers = ["desired", "current"]
            # print(pd.DataFrame(data, index=row_names, columns=headers))
            
            drone_simulator.sim_step(
                desired_thrust, roll_thrust=roll_thrust,
                pitch_thrust=pitch_thrust, yaw_thrust=yaw_thrust
            )

        current_pos, _ = drone_simulator.position_sensor()
        assert np.linalg.norm(np.array(current_pos) - np.array(pos_targets[-1])) < 0.2, "Drone did not reach the final target!"
        print(f"Task ({task_label}) completed successfully!")
    finally:
        try:
            view.close()
        except Exception:
            pass


def main(
    wind: bool = False,
    rotated_gates: bool = False,
    all_tasks: bool = False,
    runs: int = 10,
    rendering_freq: float = 3.0,
    fixed_track: bool = False,
) -> None:
    task_list = []
    if all_tasks:
        task_list = [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
    else:
        task_list = [(wind, rotated_gates)]

    for wind_flag, rotated in task_list:
        for run_idx in range(runs):
            print(f"\nRun {run_idx + 1}/{runs} for wind={wind_flag}, rotated_gates={rotated}")
            run_single_task(
                wind=wind_flag,
                rotated_gates=rotated,
                rendering_freq=rendering_freq,
                fixed_track=fixed_track,
            )


if __name__ == '__main__':
    tyro.cli(main)