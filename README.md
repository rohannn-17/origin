# Origin Assignment - Custom DWA Local Planner (ROS2 Humble)

## Objective

The objective of this assignment is to implement a **custom Dynamic Window Approach (DWA) local planner** from scratch in ROS2 Humble for a TurtleBot3 simulated in Gazebo.

The planner must:

- Sample velocity commands within dynamic constraints
- Predict trajectories for each sampled velocity pair
- Evaluate trajectories using a cost function:
  - Distance to goal
  - Obstacle avoidance
  - Path smoothness
- Publish optimal `/cmd_vel`
- Visualize trajectories using RViz Markers
- Log meaningful debugging information

No use of `nav2_dwb_controller` or prebuilt planners.

Note: Goal/Target-point is toggling between (x,y) = (1.3, 1.3) & ((-1.3, -1.3)

---

# Algorithm Overview

The implemented planner follows the Dynamic Window Approach (DWA).

At every control cycle:

1. Read robot pose from `/odom`
2. Read obstacle data from `/scan`
3. Compute dynamic velocity window based on current velocity and acceleration limits
4. Sample candidate `(v, w)` pairs
5. Simulate trajectory for each candidate
6. Compute cost for each trajectory
7. Select best velocity command
8. Publish `/cmd_vel`
9. Visualize trajectories and path history in RViz

---

## Visualization in RViz

The planner publishes two marker types:

1. **Sampled DWA trajectories** → Green line segments  
2. **Robot path history** → Red continuous line strip  

This allows visualization of:
- The entire decision space at each timestep
- The optimal chosen trajectory
- The historical path taken by the robot

---

## RViz Setup Instructions

### 1. Launch TurtleBot3 Simulation

```bash
source /opt/ros/humble/setup.bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### 2. Run the DWA Planner

```bash
cd ~/dwa_ws
colcon build
source install/setup.bash
ros2 run dwa_local_planner dwa_node
```

### 3. Start RViz

```bash
rviz2
```

### 4. Configure RViz

- Set Fixed Frame → odom
- Add → Marker
  - Topic → /dwa_trajectories
- Add → LaserScan
  - Topic → /scan
- Add → TF

You should now see:
- Sampled trajectory rollouts
- Robot path history trail
- Live LiDAR scan
- Robot navigating toward goal
---

# Pseudocode

```text
INITIALIZE:
    subscribe to /odom
    subscribe to /scan
    publish to /cmd_vel
    publish to /dwa_trajectories (Markers)

LOOP every dt seconds:

    if odom or scan not available:
        return

    read current pose (x, y, yaw)
    read current velocities (v, w)

    compute distance to goal

    if goal reached:
        flip goal coordinates
        return

    compute dynamic window:
        v_min = max(min_v, v - acc_v * dt)
        v_max = min(max_v, v + acc_v * dt)
        w_min = max(-max_w, w - acc_w * dt)
        w_max = min(max_w, w + acc_w * dt)

    sample velocity candidates:
        v in [v_min, v_max]
        w in [-max_w, max_w]

    best_cost = infinity

    for each (v_s, w_s):
        simulate trajectory using unicycle model:
            x += v * cos(yaw) * dt
            y += v * sin(yaw) * dt
            yaw += w * dt

        compute:
            goal_cost = distance from trajectory endpoint to goal
            obs_cost = inverse of min distance to obstacles
            smooth_cost = |w|

        total_cost = 
            w_goal * goal_cost +
            w_obs * obs_cost +
            w_smooth * smooth_cost

        if collision:
            discard trajectory

        if total_cost < best_cost:
            best_cmd = (v_s, w_s)

    publish best_cmd to /cmd_vel
    publish sampled trajectories to RViz
    append current position to path history
```
---

### Components

- **Goal Cost** → Euclidean distance from trajectory endpoint to goal  
- **Obstacle Cost** → Penalizes trajectories that pass close to obstacles  
- **Smoothness Cost** → Penalizes high angular velocity to avoid jerky motion  
- **Collision Handling** → Any trajectory intersecting the robot radius is rejected (assigned very high cost)

### Software Versions used

| Software   | Version           |
| ---------- | ----------------- |
| Ubuntu     | 22.04 LTS         |
| ROS2       | Humble Hawksbill  |
| Gazebo     | Fortress 6.16.0   |
| TurtleBot3 | Burger            |
| Python     | 3.10              |
| NumPy      | 1.24+             |
| RViz2      | Humble Compatible |

