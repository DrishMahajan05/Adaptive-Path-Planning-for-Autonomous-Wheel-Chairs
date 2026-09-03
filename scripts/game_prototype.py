import pygame
import numpy as np
import math

# --- Constants & Hyperparameters ---
TAU_C = 3.0       # Time delay constant
T_A = 0.4         # Approaching speed factor
K_MAX = 10.0      # Max angular rate gain
A_Y_UB = 0.1      # Max lateral acceleration limit (m/s^2)
V_MAX = 1.0       # Max linear speed (m/s)
WHEEL_DIST = 0.5  # Distance between wheels (d)

class Wheelchair:
    def __init__(self, x, y):
        # State variables
        self.x = x
        self.y = y
        self.theta = 0.0
        self.v = 0.0      # Current linear speed
        self.omega = 0.0  # Current angular speed
        
        # Desired states
        self.v_d = 0.0
        self.theta_d = 0.0
        
    def update_WAP(self, target_x, target_y):
        """
        Waypoint and Attitude Planning: 
        Calculates the desired angle to the next user-drawn point.
        """
        dx = target_x - self.x
        dy = target_y - self.y
        self.theta_d = math.atan2(dy, dx)
        
        # Distance to target
        return math.hypot(dx, dy)

    def update_SPD(self, distance_to_target):
        """
        Speed Profile Design:
        Simplification of the bang-bang control optimization.
        Accelerates if far, decelerates if close.
        """
        if distance_to_target > 20:
            self.v_d = V_MAX
        else:
            self.v_d = max(0.0, (distance_to_target / 20.0) * V_MAX)

    def update_ARGA(self):
        """
        Angular Rate Gain Adaptation:
        Caps the turning speed based on current linear speed to ensure patient comfort.
        """
        angle_diff = (self.theta_d - self.theta + math.pi) % (2 * math.pi) - math.pi
        
        # Prevent division by zero
        if abs(angle_diff) < 0.01 or self.v < 0.01:
            K = K_MAX
        else:
            # Paper's ARGA formula implementation
            denominator = self.v * abs(angle_diff) * (1 - math.exp(-T_A / TAU_C))
            K_calculated = A_Y_UB / denominator if denominator != 0 else K_MAX
            K = min(K_calculated, K_MAX)
            
        self.omega = K * angle_diff * (1 - math.exp(-T_A / TAU_C))

    def update_mHRVO(self, obstacles):
        """
        Placeholder for modified Hybrid Reciprocal Velocity Obstacle.
        In a full implementation, this checks predicted future positions 
        against obstacle velocities and modifies v_d and theta_d.
        """
        pass # To be implemented with an RVO library or complex geometric checks

    def step(self, dt):
        """
        First-order dynamic system approximation for kinematics.
        """
        # Linear speed approaches desired speed with time delay TAU_C
        self.v += (-1/TAU_C * self.v + 1/TAU_C * self.v_d) * dt
        
        # Update poses based on two-wheel kinematics
        self.theta += self.omega * dt
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

robot = Wheelchair(400, 300)
waypoints = []

running = True
while running:
    dt = clock.tick(60) / 1000.0  # Delta time in seconds

    # 1. Handle user input (drawing paths)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEMOTION:
            # Add waypoints when mouse is clicked and dragged
            if pygame.mouse.get_pressed()[0]:  
                waypoints.append(event.pos)

    # 2. Run Algorithm Blocks
    if waypoints:
        target_x, target_y = waypoints[0]
        
        # Run WAP
        dist = robot.update_WAP(target_x, target_y)
        
        # Pop waypoint if reached
        if dist < 5.0:
            waypoints.pop(0)
            
        # Run SPD, ARGA, mHRVO
        robot.update_SPD(dist)
        robot.update_ARGA()
        robot.update_mHRVO(obstacles=[]) # No dynamic obstacles yet
    else:
        robot.v_d = 0.0 # Stop if no path

    # Update kinematics
    robot.step(dt)

    # 3. Draw Everything
    screen.fill((255, 255, 255))
    
    # Draw path
    if len(waypoints) > 1:
        pygame.draw.lines(screen, (200, 200, 200), False, waypoints, 2)
        
    # Draw wheelchair
    pygame.draw.circle(screen, (0, 0, 255), (int(robot.x), int(robot.y)), 10)
    end_x = robot.x + 15 * math.cos(robot.theta)
    end_y = robot.y + 15 * math.sin(robot.theta)
    pygame.draw.line(screen, (255, 0, 0), (int(robot.x), int(robot.y)), (int(end_x), int(end_y)), 3)

    pygame.display.flip()

pygame.quit()
