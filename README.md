# Autonomous Quadcopter Control in MuJoCo 🚁 

This project implements a robust cascade PID control system for an autonomous quadcopter simulation using the MuJoCo physics engine. The drone navigates through a sequence of gates (including rotated ones) and maintains stability under stochastic wind conditions.

![Simulation Demo](demo.gif)
*(Placeholder for your simulation screenshot or GIF)*

## 🚀 Key Features

* **Cascade PID Control:** Separate loops for Position (Outer) and Attitude (Inner) stabilization.
* **Robust Navigation:** Implemented "Nose-in" orientation logic — the drone always faces its flight direction.
* **Coordinate Transformation:** Real-time transformation of global position errors into the drone's **Body Frame** using rotation matrices. This ensures correct pitch/roll commands regardless of the drone's yaw.
* **Wind Compensation:** Integral (I) terms in the position controller allow the drone to "lean" into the wind and maintain position against external forces.
* **Derivative-on-Measurement:** PID implementation uses derivative on measurement (instead of error) to prevent "derivative kick" when target waypoints change abruptly.