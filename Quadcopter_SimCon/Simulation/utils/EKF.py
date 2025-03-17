import numpy as np
import pdb

class EKF_IMU:
    def __init__(self, dt, process_noise, measurement_noise):
        self.dt = dt  # Time step

        # State vector [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.x = np.zeros((6, 1))

        # State covariance matrix
        self.P = np.eye(6) * 0.1  

        # Process noise covariance
        self.Q = np.eye(6) * process_noise  

        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise  

        # State transition matrix (Linearized)
        self.F = np.eye(6)
        self.F[:3, 3:] = np.eye(3) * dt  # Position update based on velocity

        # Measurement matrix (Assuming accel/mag measure only angles)
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))

    def predict(self, gyro_data):
        """ Prediction step using gyroscope data """
        # Gyro readings as control input
        u = np.array([[gyro_data[0]], [gyro_data[1]], [gyro_data[2]],
                      [0], [0], [0]])  

        # Predict next state
        self.x = self.F @ self.x + u * self.dt  

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
          

    def update(self, accel_data, mag_data):
        """ Update step using accelerometer and magnetometer """
        # Convert accelerometer to roll, pitch
        roll_meas = np.arctan2(accel_data[1], -accel_data[2])
        pitch_meas = np.arctan2(-accel_data[0], np.sqrt(accel_data[1]**2 + accel_data[2]**2))

        # Convert magnetometer to yaw
        yaw_meas = np.arctan2(mag_data[1], mag_data[0])

        # Measurement vector
        z = np.array([[roll_meas], [pitch_meas], [yaw_meas]])

        # Innovation (measurement residual)
        y = z - self.H @ self.x  

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R  

        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)  

        # State update
        self.x = self.x + K @ y  

        # Covariance update
        self.P = (np.eye(6) - K @ self.H) @ self.P  

    def get_orientation(self):
        """ Returns roll, pitch, yaw (in rads) """
        return self.x[:3].flatten()
