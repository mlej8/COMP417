#!/usr/bin/python

__author__ = "Travis Manderson"
__copyright__ = "Copyright 2018, Travis Manderson"

#Implement the ball tracking using OpenCV
import simulation
import pid_plotter_vispy
from multiprocessing import Process, Manager, Lock
import multiprocessing

__author__ = "Travis Manderson"
__copyright__ = "Copyright 2018, Travis Manderson"

from helpers import *
import simulation

import random
import cv2
import math
import numpy as np
import csv
import os, sys
import pygame
import pickle
import time

from PID_variables import *
from opencv_ball_tracker import PID_controller
class PID_controller_kf(PID_controller):
    def __init__(self, target_pos = -0.1):
        #TODO YOU CAN ADD new variables as needed
        super().__init__(target_pos, use_kf=True)
        
        # parameters for the kalman filter
        self.motion_sig = 1.0            # ball motion variance 
        self.measurement_sig = 1.0       # measurement variance (uncertainty)
        self.mu = 0.0                    # gaussian mean (estimate mean)
        self.sig = 10.                # gaussian variance (estimate uncertainty)

        self.Kp = kf_Kp
        self.Ki = kf_Ki
        self.Kd = kf_Kd
        self.bias = kf_bias

    def update_state(self, measurement): 
        """ State Update Equation """
        K_gain = self.compute_kalman_gain()
        # self.mu contains the predicted value from previous state of the current state
        # update the current state with the measurement and the predicted value
        self.mu = (1 - K_gain) * self.mu + K_gain * measurement

        # update state estimate uncertainty
        self.update_estimate_uncertainty()

    def predict(self):
        """ Predict next state """
        # get the motion of the ball (assuming 1D for Kalman Filter as x never changes)
        motion = self.mu - self.last_pos

        # Dynamic model: past position + some linear movement
        self.mu += motion

        # predicted new position is noisy
        self.sig += self.motion_sig

    def compute_kalman_gain(self):
        K_gain = (self.sig / (self.sig + self.measurement_sig))
        return K_gain
    
    def update_estimate_uncertainty(self):
        """Covariance update Equation
        
        Update the estimate uncertainty of the current state.
        We can see from the equation that the estimate uncertainty always gets smaller with each filter iteration, because (1 - K_gain) <= 1.
        If measurement uncertainty is large, K_gain will be small, consequently convergence (to 0) of estimate uncertainty will be slow.
        If measurement uncertainty is small, K_gain will be large, consequently convergence (to 0) of estimate uncertainty will be fast.
        """
        self.sig = (1 - self.compute_kalman_gain()) * self.sig

    def get_kalman_filter_estimation(self, observation):
        #TODO Implement the kalman filter ... 
        # observation is the estimated x,y position of the detect image

        self.update_state(observation[1])
        
        # return the kalman filter adjusted values
        kalman_filter_estimate = [observation[0], self.mu] 

        self.predict()
        return kalman_filter_estimate[0], kalman_filter_estimate[1]

def run_simulator(graph_index, graph_time, graph_position, graph_error, graph_fan, graph_target, validation_mode, save_mode, noisy_mode):
    global xdata
    global ydata
    env = simulation.env(PID_controller_kf, graph_index, graph_time, graph_position, graph_error, graph_fan, graph_target)
    if validation_mode:
        env.run_validation(noisy_mode, save_mode)
    else:
        env.run(noisy_mode)

def run_pid_plotter(graph_index, graph_time, graph_position, graph_error, graph_fan, graph_target, headless_mode):
    try:
        plotter = pid_plotter_vispy.PIDPlotter(graph_index, graph_time, graph_position, graph_error, graph_fan, graph_target)
        plotter.run()
    except Exception as e:
        if not headless_mode:
            print(e)
    return

if __name__ == '__main__':
    save_mode = False
    headless_mode = False
    if len(sys.argv) > 1:
        if sys.argv[1] == 'headless':
            print("Headless mode activated. ")
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            pygame.init()
            pygame.display.set_mode((1, 1))
            save_mode = True
            headless_mode = True

    print("welcome to the red ball simulator")
    validation_mode = False
    noisy_mode = False
    exit = False
    while not exit:
        print('v - Validation Mode')
        print('vn - Validation Noisy Mode')

        print('vs - Validation Save Video Mode')
        print('quit - Exit')

        inputString = input("Select Job To Run: ")
        # inputString = "v 0.5"
        # inputString = "e"
        commands = inputString.split(";")
        for command in commands:
            argList = command.strip()
            argList = argList.split(" ")
            job = argList[0]
            if job == 'v' or job == 'vs' or job == 'vn':
                validation_mode = True
            if job == 'vs':
                save_mode = True

            if job == "vn":
                noisy_mode = True

            if job == "quit":
                quit()

            if job == 'v' or job == 'vs' or job=='vn':
                # graph_lock = Lock()
                max_size = 432000 #1 hour at 120 fps
                graph_time = multiprocessing.Array('d', max_size)
                graph_position = multiprocessing.Array('d', max_size)
                graph_error = multiprocessing.Array('d', max_size)
                graph_fan = multiprocessing.Array('d', max_size)
                graph_target = multiprocessing.Array('d', max_size)
                graph_index = multiprocessing.Value('i')
                graph_index.value = 0

                processes = []
                process_plotter = Process(target=run_pid_plotter,
                                          args=(graph_index, graph_time, graph_position, graph_error, graph_fan, graph_target, headless_mode))
                process_plotter.start()
                processes.append(process_plotter)

                process_sim = Process(target=run_simulator, args=(graph_index, graph_time, graph_position, graph_error, graph_fan, graph_target, validation_mode, save_mode, noisy_mode))
                process_sim.start()
                processes.append(process_sim)

                try:
                    for process in processes:
                        process.join()
                except Exception as e:
                    print("test")
                    print(e)
                print("Exiting Main Thread")

