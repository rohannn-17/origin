#!/usr/bin/env python3
import math
import threading
import sys
import select
import termios
import tty
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from tf_transformations import euler_from_quaternion


class DWALocalPlanner(Node):

    def __init__(self):
        super().__init__('dwa_local_planner_fixed')

        # Publishers / Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/dwa_trajectories', 10)

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Control loop
        self.dt = 0.1
        self.timer = self.create_timer(self.dt, self.control_loop)

        # State
        self.odom = None
        self.scan = None

        # Goal
        self.goal_x = 1.5
        self.goal_y = 1.5
        self.goal_reached_thresh = 0.05

        # Robot limits
        self.max_v = 0.15
        self.min_v = 0.0
        self.max_w = 2.0

        self.max_acc_v = 0.5
        self.max_acc_w = 2.0

        # DWA sampling
        self.v_samples = 6
        self.w_samples = 20
        self.predict_time = 3.0

        # Cost weights
        self.w_goal = 3.0
        self.w_obs = 1.5
        self.w_smooth = 0.1

        # Obstacle thresholds
        self.robot_radius = 0.18
        self.min_prefer_dist = 0.5
        self.min_safe_dist = 0.05

        # Oscillation detection (history of recent chosen commands)
        self.angular_history = deque(maxlen=16)  # recent angular commands (signed)
        self.linear_history = deque(maxlen=16)   # recent linear commands
        self.oscillation_detected = False

        # Rotation-then-move state
        self.need_rotate_then_move = False
        self.rotate_target_yaw = None
        self.allow_forward_towards_preferred = False
        self.forward_prefer_count = 0
        self.preferred_heading = None
        self.pref_heading_weight = 2.0  # penalty weight for deviating from preferred heading

        self.get_logger().info("DWA planner started. Goal=(%.2f, %.2f)" % (self.goal_x, self.goal_y))

    # -------------------- Callbacks --------------------

    def odom_callback(self, msg):
        self.odom = msg

    def scan_callback(self, msg):
        self.scan = msg

    # -------------------- Main Control Loop --------------------

    def control_loop(self):
        # need sensors
        if self.odom is None or self.scan is None:
            return

        x, y, yaw, v_cur, w_cur = self.get_state()


        dist_to_goal = math.hypot(self.goal_x - x, self.goal_y - y)

        # Goal reached -> flip and reset rotation/move state
        if dist_to_goal < self.goal_reached_thresh:
            self.goal_x = -self.goal_x
            self.goal_y = -self.goal_y
            self.need_rotate_then_move = False
            self.allow_forward_towards_preferred = False
            self.forward_prefer_count = 0
            self.rotate_target_yaw = None
            self.preferred_heading = None
            self.angular_history.clear()
            self.linear_history.clear()
            self.get_logger().info("Goal reached. Flipped to (%.2f, %.2f)" % (self.goal_x, self.goal_y))
            return

        # # If currently in rotate-then-move, perform rotation until aligned
        # if self.need_rotate_then_move and self.rotate_target_yaw is not None:
        #     angle_to_target = self.angle_diff(self.rotate_target_yaw, yaw)
        #     if abs(angle_to_target) > 0.08:  # ~4.5 deg tolerance
        #         cmd = Twist()
        #         cmd.linear.x = 0.0
        #         ang_speed = max(0.25, min(self.max_w, 2.0 * abs(angle_to_target)))
        #         cmd.angular.z = math.copysign(min(ang_speed, self.max_w), angle_to_target)
        #         self.cmd_vel_pub.publish(cmd)
        #         self.get_logger().info(f"[ROTATING_RECOVERY] angle_err={angle_to_target:.2f}, w={cmd.angular.z:.2f}")
        #         # record history for oscillation detection
        #         self.angular_history.append(cmd.angular.z)
        #         self.linear_history.append(cmd.linear.x)
        #         return
        #     else:
        #         # rotation finished — enable forward preference for a few cycles
        #         self.need_rotate_then_move = False
        #         self.allow_forward_towards_preferred = True
        #         self.forward_prefer_count = 20
        #         self.preferred_heading = self.rotate_target_yaw
        #         self.rotate_target_yaw = None
        #         self.get_logger().info("[RECOVERY] rotation done; will prefer forward toward chosen heading")
        #         # fall through to planning this cycle

        # decrement preference counter
        if self.allow_forward_towards_preferred and self.forward_prefer_count > 0:
            self.forward_prefer_count -= 1
        if self.forward_prefer_count <= 0:
            self.allow_forward_towards_preferred = False
            self.preferred_heading = None

        # compute heading error
        goal_angle = math.atan2(self.goal_y - y, self.goal_x - x)
        angle_diff = self.angle_diff(goal_angle, yaw)

        # dynamic window
        v_min, v_max, w_min, w_max = self.compute_dynamic_window(v_cur, w_cur)
        # allow full angular exploration
        w_min = -self.max_w
        w_max = self.max_w

        # sampling candidates
        v_candidates = np.linspace(v_min, v_max, self.v_samples)
        w_candidates = np.linspace(w_min, w_max, self.w_samples)

        best_cost = float('inf')
        best_cmd = None
        best_components = None
        best_traj_info = None

        # track rotation-only best (clearance)
        best_rotation_candidate = None
        best_rotation_min_dist = -1.0

        trajectories_for_vis = []

        # Evaluate DWA candidates
        for v_s in v_candidates:
            for w_s in w_candidates:
                traj, end_yaw = self.rollout_trajectory(x, y, yaw, v_s, w_s)
                total_cost, components, min_dist = self.evaluate_trajectory_components(traj, end_yaw, v_s, w_s)

                # If we are preferring a heading, bias forward v>0 candidates to that heading
                if self.allow_forward_towards_preferred and v_s > 0 and self.preferred_heading is not None:
                    heading_err = abs(self.angle_diff(end_yaw, self.preferred_heading))
                    total_cost += self.pref_heading_weight * heading_err

                trajectories_for_vis.append(traj)

                # rotation-only candidate bookkeeping (v==0)
                if abs(v_s) < 1e-6:
                    if min_dist > best_rotation_min_dist:
                        best_rotation_min_dist = min_dist
                        best_rotation_candidate = (v_s, w_s, end_yaw, min_dist)

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_cmd = (v_s, w_s)
                    best_components = components
                    best_traj_info = (traj, end_yaw, min_dist)

        # Publish visualization
        self.publish_trajectories(trajectories_for_vis)

        # record candidate in history for oscillation detection (we record what we'd command)
        if best_cmd is not None:
            self.angular_history.append(best_cmd[1])
            self.linear_history.append(best_cmd[0])


        # choose and publish best command from DWA normally
        cmd = Twist()
        if best_cmd is not None and best_cost < 1e9:
            cmd.linear.x = float(best_cmd[0])
            cmd.angular.z = float(best_cmd[1])
            self.get_logger().info(
                "[AUTO] v={:.3f}, w={:.3f}, total={:.3f}, goal={:.3f}, obs={:.3f}, smooth={:.3f}, angle_err={:.2f}, goal coor= {:.3f},{:.3f}, coor={:.3f},{:.3f}, dist to goal= {:.3f}".format(
                    cmd.linear.x, cmd.angular.z, best_cost,
                    best_components[0], best_components[1], best_components[2], angle_diff, self.goal_x, self.goal_y,x,y,dist_to_goal
                )
            )
        else:
            # fallback
            cmd = self.pursue_goal(x, y, yaw)
            self.get_logger().warn("[AUTO] No safe DWA candidate -> fallback pursuit")

        # Emergency stop if obstacle extremely close
        min_obs = self.compute_min_obstacle_distance()
        if min_obs < self.min_safe_dist:
            self.get_logger().warn(f"EMERGENCY: obstacle {min_obs:.2f}m -> stop")
            cmd.linear.x = 0.0
            if abs(cmd.angular.z) < 0.1:
                cmd.angular.z = math.copysign(0.8, angle_diff)

        self.cmd_vel_pub.publish(cmd)

    # -------------------- Helpers (DWA) --------------------

    def get_state(self):
        pos = self.odom.pose.pose.position
        ori = self.odom.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        v = self.odom.twist.twist.linear.x
        w = self.odom.twist.twist.angular.z
        return pos.x, pos.y, yaw, v, w

    def compute_dynamic_window(self, v, w):
        v_min = max(self.min_v, v - self.max_acc_v * self.dt)
        v_max = min(self.max_v, v + self.max_acc_v * self.dt)
        w_min = max(-self.max_w, w - self.max_acc_w * self.dt)
        w_max = min(self.max_w, w + self.max_acc_w * self.dt)
        return v_min, v_max, w_min, w_max

    def rollout_trajectory(self, x, y, yaw, v, w):
        """
        Returns: (list of (x,y) points, end_yaw)
        """
        traj = []
        t = 0.0
        x_i, y_i, yaw_i = x, y, yaw
        while t <= self.predict_time:
            x_i += v * math.cos(yaw_i) * self.dt
            y_i += v * math.sin(yaw_i) * self.dt
            yaw_i += w * self.dt
            # normalize yaw occasionally
            if yaw_i > math.pi or yaw_i < -math.pi:
                yaw_i = (yaw_i + math.pi) % (2.0 * math.pi) - math.pi
            traj.append((x_i, y_i))
            t += self.dt
        return traj, yaw_i

    # -------------------- Cost / Evaluation --------------------

    def evaluate_trajectory_components(self, traj, end_yaw, v, w):
        """
        Returns:
          total_cost (float),
          (goal_cost, obs_cost, smooth_cost),
          min_dist (float)
        """
        last_x, last_y = traj[-1]
        goal_cost = math.hypot(self.goal_x - last_x, self.goal_y - last_y)

        min_dist = self.min_distance_traj_to_scan(traj)

        # collision check
        if min_dist < (self.robot_radius * 0.9):
            return 1e9, (goal_cost, 1.0, abs(w)), min_dist

        # bounded obstacle cost in [0,1)
        if min_dist >= self.min_prefer_dist:
            obs_cost = 0.0
        else:
            obs_cost = (self.min_prefer_dist - min_dist) / (self.min_prefer_dist + 1e-6)

        smooth_cost = abs(w)

        total = self.w_goal * goal_cost + self.w_obs * obs_cost + self.w_smooth * smooth_cost

        return total, (goal_cost, obs_cost, smooth_cost), min_dist

    def min_distance_traj_to_scan(self, traj):
        if self.scan is None or self.odom is None:
            return float('inf')

        ranges = np.array(self.scan.ranges, dtype=float)
        angles = np.arange(len(ranges)) * self.scan.angle_increment + self.scan.angle_min
        finite_mask = np.isfinite(ranges) & (ranges > 0.01)
        if not np.any(finite_mask):
            return float('inf')

        ranges = ranges[finite_mask]
        angles = angles[finite_mask]

        rx, ry, ryaw, _, _ = self.get_state()

        sx = rx + ranges * np.cos(ryaw + angles)
        sy = ry + ranges * np.sin(ryaw + angles)
        scan_pts = np.vstack((sx, sy)).T  # (N,2)

        traj_pts = np.array(traj, dtype=float)  # (M,2)
        if traj_pts.size == 0 or scan_pts.size == 0:
            return float('inf')

        dists = np.sqrt(((traj_pts[:, None, :] - scan_pts[None, :, :]) ** 2).sum(axis=2))
        return float(np.min(dists))

    def compute_min_obstacle_distance(self):
        if self.scan is None:
            return float('inf')
        ranges = np.array(self.scan.ranges, dtype=float)
        finite = ranges[np.isfinite(ranges)]
        if finite.size == 0:
            return float('inf')
        return float(np.min(np.clip(finite, 0.0, 10.0)))

    # -------------------- Oscillation detection & clearance selection --------------------

    def detect_oscillation(self):
        """
        Detect oscillation when:
          - recent linear velocities are near zero (robot not translating)
          - recent angular commands frequently change sign (back-and-forth)
        """
        if len(self.angular_history) < 8:
            return False
        # average absolute linear speed small?
        avg_lin = float(np.mean(np.abs(self.linear_history))) if len(self.linear_history) > 0 else 0.0
        if avg_lin > 0.03:
            return False

        # compute sign changes in angular history
        signs = [math.copysign(1.0, a) if abs(a) > 1e-3 else 0.0 for a in self.angular_history]
        # compress zeros by replacing with previous non-zero sign (to avoid false breaks)
        last = signs[0]
        for i in range(len(signs)):
            if signs[i] == 0.0:
                signs[i] = last
            else:
                last = signs[i]
        sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        # oscillation if many sign changes
        if sign_changes >= 4:
            return True
        return False

    def select_heading_with_max_clearance(self, current_yaw, num_samples=16, min_range_threshold=0.15):
        """
        Sample orientations around robot (relative angles) and pick the heading
        with the largest forward clearance according to the current LaserScan.
        Returns chosen relative angle (radians) or None if none viable.
        """
        if self.scan is None:
            return None

        # desired sample set in robot frame
        rel_angles = np.linspace(-math.pi, math.pi, num_samples, endpoint=False)

        best_rel = None
        best_range = -1.0

        # precompute scan parameters
        angle_min = self.scan.angle_min
        angle_inc = self.scan.angle_increment
        n = len(self.scan.ranges)

        for rel in rel_angles:
            # find corresponding index in scan (rel is robot-frame angle; scan angle is robot-frame as well)
            idx = int(round((rel - angle_min) / (angle_inc + 1e-12)))
            if idx < 0 or idx >= n:
                # wrap into valid index by modulo (since scans often cover -pi..pi)
                idx = idx % n
            r = self.scan.ranges[idx]
            if not math.isfinite(r):
                r = 0.0
            # small margin: prefer headings with range > min_range_threshold
            if r > best_range:
                best_range = float(r)
                best_rel = float(rel)

        if best_rel is None:
            return None

        # if best_range is too small, still return it (we will rotate and attempt to clear)
        return best_rel

    # -------------------- Pursuit Fallback --------------------

    def pursue_goal(self, x, y, yaw):
        dx = self.goal_x - x
        dy = self.goal_y - y
        angle_to_goal = math.atan2(dy, dx)
        angle_diff = self.angle_diff(angle_to_goal, yaw)

        k_lin = 0.6
        k_ang = 1.5

        v_cmd = max(0.0, min(self.max_v, k_lin * math.hypot(dx, dy)))
        w_cmd = max(-self.max_w, min(self.max_w, k_ang * angle_diff))

        cmd = Twist()
        if abs(angle_diff) > 0.4:
            cmd.linear.x = 0.0
            cmd.angular.z = w_cmd
        else:
            cmd.linear.x = v_cmd
            cmd.angular.z = w_cmd

        if self.compute_min_obstacle_distance() < 0.15:
            cmd.linear.x = 0.0

        return cmd

    def angle_diff(self, a, b):
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    # -------------------- Visualization --------------------

    def publish_trajectories(self, trajectories):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "dwa"
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.01
        m.color.a = 1.0
        m.color.g = 1.0
        m.color.r = 0.0

        m.points = []
        for traj in trajectories:
            for i in range(len(traj) - 1):
                p1 = Point()
                p1.x = float(traj[i][0]); p1.y = float(traj[i][1]); p1.z = 0.0
                p2 = Point()
                p2.x = float(traj[i+1][0]); p2.y = float(traj[i+1][1]); p2.z = 0.0
                m.points.append(p1); m.points.append(p2)

        self.marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = DWALocalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_vel_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
