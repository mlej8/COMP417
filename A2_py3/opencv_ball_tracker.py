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

class PID_controller:
    def __init__(self, target_pos = -0.1):
        # rescale image
        def rescale(pixel_value, a, b, img_min, img_max):
            return (((b-a) * (pixel_value - img_min)) / (img_max - img_min)) + a
        self.rescale = np.vectorize(rescale)
        
        ###EVERYTHING HERE MUST BE INCLUDED###
        self.target_pos = target_pos

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.bias = bias

        self.detected_pos = 0.0
        self.rpm_output = 0.0
        ######################################

        self.error_pos = 0.0
        self.last_pos = 0.0
        self.last_error_pos = 0.0
        self.acc_pos_error = 0.0

        self.last_t = None

        self.min = 1000000
        self.max = 0
        return

    def set_target(self, target_pos):
        self.target_pos = target_pos

    def reset(self):
        self.detected_pos = 0.0
        self.rpm_output = 0.0
        self.error_pos = 0.0
        self.last_pos = 0.0
        self.last_error_pos = 0.0
        self.acc_pos_error = 0.0
        self.last_t = None
        return

    def detect_ball(self, frame):
        #TODO
        #You are given a basic opencv ball tracker. However, this won't work well for the noisy case.
        #Play around to get it working.

        bgr_color = 31, 0, 142

        # denoising (gaussian, median, non-local)
        """ 
        Kernel is normalized by KERNEL_SIZE^2. 
        This makes all elements in the matrix sum up to 1.
        We don't want to change the energy of the original image, and by multiplying the image by 1, we are not changing the total energy contained in the image. 
        """
        KERNEL_SIZE = 5
        kernel = np.ones((KERNEL_SIZE,KERNEL_SIZE), np.float32) / KERNEL_SIZE**2 # float for more precision on the image normalized by 25
        # denoised_img = cv2.filter2D(frame, -1, kernel)
        # cv2.imwrite("test/original_img.jpg", frame)
        # cv2.imwrite(f"test/denoise_img_kernel_{KERNEL_SIZE}.jpg", denoised_img)
        # denoised_blur_img = cv2.blur(frame, (KERNEL_SIZE, KERNEL_SIZE))
        # cv2.imwrite(f"test/denoise_img_blur_{KERNEL_SIZE}.jpg", denoised_blur_img)
        # median_img = cv2.medianBlur(frame, KERNEL_SIZE)
        # cv2.imwrite(f"test/denoise_img_median_blur_{KERNEL_SIZE}.jpg", median_img)
        # bilateral_img = cv2.bilateralFilter(frame, 9, 75, 75)
        # cv2.imwrite(f"test/denoise_img_bilateral_blur_{KERNEL_SIZE}.jpg", bilateral_img)
        # frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,15)
        # cv2.imwrite(f"test/denoise_img_nlm_blur_{KERNEL_SIZE}.jpg", nlm)

        for i in range(8):
            frame = cv2.filter2D(frame, -1, kernel)
            # frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,15)
            # frame = cv2.GaussianBlur(frame, (KERNEL_SIZE, KERNEL_SIZE), 0)
            # cv2.imwrite(f"test/denoise_img_{KERNEL_SIZE}_{i}.jpg", frame)
            # cv2.imwrite(f"test/nlm_img_{KERNEL_SIZE}_{i}.jpg", frame)
            # cv2.imwrite(f"test/gaussian_img_{KERNEL_SIZE}_{i}.jpg", frame)
        
        
        frame = self.rescale(frame, 0, 255, frame.min(), frame.max()).astype("uint8")
        # frame = self.rescale(nlm, 0, 255, frame.min(), frame.max()).astype("uint8")
        cv2.imwrite(f"test/rescaled_img_{KERNEL_SIZE}.jpg", frame)

        thresh = 100
        hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        HSV_lower = np.array([hsv_color[0] - thresh, hsv_color[1] - thresh, hsv_color[2] - thresh])
        HSV_upper = np.array([hsv_color[0] + thresh, hsv_color[1] + thresh, hsv_color[2] + thresh])

        x, y, radius = -1, -1, -1
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imwrite(f"test/hsv_frame.jpg", hsv_frame)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        # mask = cv2.inRange(hsv, color_lower, color_upper)
        # mask = cv2.inRange(hsv_frame, HSV_lower, HSV_upper)
        mask = cv2.inRange(hsv_frame, HSV_lower, HSV_upper)
        cv2.imwrite(f"test/mask.jpg", mask)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0]
        center = (-1, -1)
        # only proceed if at least one contour was found
        try:
            if len(contours) > 0:
                print("Found contour")
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(mask)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                #Ball is lost(went above the camera's line of sight)
                if radius <= 2:
                    return -1, -1
            else: # TODO erase this
                print("nothing")
        except e:
            print("no contour found ...")
            center = (-1, -1)
            pass
        return center[0], center[1]  # x, y , radius


    # def get_fan_rpm(self, image_frame=None, position=None):
    def get_fan_rpm(self, image_frame=None, position=None):
        #TODO Get the FAN RPM to push the ball to the target position
        #The slide moving up and down is where the ball is supposed to be
        pos = 0.0
        if image_frame is not None:
            x, y = self.detect_ball(image_frame)
            if y >= 0:
                if y > self.max:
                    self.max = y
                if y < self.min:
                    self.min = y
            # print("min: {}, max: {}".format(self.min, self.max))
            # range is: 165 - 478, or 600-y is: 122 - 435
            # target is 121
            range = 485 - 0
            pos = (float(485 - y)) / (485 - 0)  # scaled position
            self.detected_pos = pos
        if position != None:
            pos = position
        output = 0.0
        p_error = 0.0
        d_error = 0.0
        i_error = 0.0

        target_vel = 0.0
        self.error_pos = self.target_pos - pos
        error_vel = 0.0
        # print('detected at: {}, {}'.format(x, y))
        t = time.time()
        fan_rpm = 0
        if self.last_t is not None:
            # TODO
            # implement the PID controller function and compute FAN rpm to reach the target position.
            # accumulate errors for the integral
            self.acc_pos_error += self.error_pos
            
            # compute the difference in position between current frame and last frame
            error_diff = self.error_pos - self.last_error_pos

            # if error_pos is negative, it means that we have to turn the fan on more
            # if error_pos is positive, it means that we have to turn the fan on less
            fan_rpm = self.Kp * self.error_pos - self.Kd * error_diff + self.Ki * self.acc_pos_error 
            print("Pos: {}", self.detected_pos)
            print("Fan RPM: {} \tError: {}\tP: {}\tD: {}\tI: {}".format(fan_rpm, self.error_pos, self.Kp * self.error_pos, -self.Kd * error_diff, self.Ki * self.acc_pos_error))
            # TODO consider using self.prev_fan_rpm + fan_rpm ?
            # TODO figure out what to do with the bias ?
            # fan_rpm =+ self.bias

        self.last_t = t
        self.last_pos = pos
        self.last_error_pos = self.error_pos
        # print('p_e: {:10.4f}, d_e: {:10.4f}, i_e: {:10.4f}, output: {:10.4f}'.format(p_error, d_error, i_error, output))
        return fan_rpm

def run_simulator(graph_index, graph_time, graph_position, graph_error, graph_fan, graph_target, validation_mode, save_mode, noisy_mode):
    global xdata
    global ydata
    env = simulation.env(PID_controller, graph_index, graph_time, graph_position, graph_error, graph_fan, graph_target)
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
            if job == 'v' or job == 'vs' or job=='vn':
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

                # while True:
                #     pass
                print("Exiting Main Thread")

