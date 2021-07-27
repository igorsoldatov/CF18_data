import pandas as pd
import numpy as np
from enum import Enum

data_dir = './raw_calib_data/'
gravity = 9.807


class AccelRange(Enum):
   ACCEL_RANGE_2G = 1
   ACCEL_RANGE_4G = 2
   ACCEL_RANGE_8G = 3
   ACCEL_RANGE_16G = 4


class GyroRange(Enum):
   GYRO_RANGE_250DPS = 1
   GYRO_RANGE_500DPS = 2
   GYRO_RANGE_1000DPS = 3
   GYRO_RANGE_2000DPS = 4


def get_gyro_scale(gyro_range):
    scale = 0.0
    if gyro_range == GyroRange.GYRO_RANGE_250DPS:
        scale = 131.0 / 250.0
    elif gyro_range == GyroRange.GYRO_RANGE_500DPS:
        scale = 65.5 / 500.0
    elif gyro_range == GyroRange.GYRO_RANGE_1000DPS:
        scale = 32.8 / 1000.0
    elif gyro_range == GyroRange.GYRO_RANGE_2000DPS:
        scale = 16.4 / 2000.0
    return scale


def get_accel_scale(accel_range):
    range_value = 0.0
    if accel_range == AccelRange.ACCEL_RANGE_2G:
        range_value = 2.0
    elif accel_range == AccelRange.ACCEL_RANGE_4G:
        range_value = 4.0
    elif accel_range == AccelRange.ACCEL_RANGE_8G:
        range_value = 8.0
    elif accel_range == AccelRange.ACCEL_RANGE_16G:
        range_value = 16.0
    scale = gravity * range_value / 32767.5
    return scale


def calibrate_gyroscope(data, config, gyro_range):
    scale = get_gyro_scale(gyro_range)
    ids = config["ids"]
    total_x = np.sum(data[:, ids[0]]) * scale
    total_y = np.sum(data[:, ids[1]]) * scale
    total_z = np.sum(data[:, ids[2]]) * scale
    counter = len(data)
    bias_x = total_x / counter
    bias_y = total_y / counter
    bias_z = total_z / counter

    params = {'scale_factor': scale,
              'bias': {'x': bias_x, 'y': bias_y, 'z': bias_z},
              'sensitivity': {'x': 1, 'y': 1, 'z': 1},
              'min': {'x': 0, 'y': 0, 'z': 0},
              'max': {'x': 0, 'y': 0, 'z': 0}}

    return params


def calibrate_accelerometer(data, start, end, config, accel_range):
    scale = get_accel_scale(accel_range)
    ids = config["ids"]

    max_val = [None, None, None]
    min_val = [None, None, None]
    avg_list = [[], [], []]

    # для каждого состояния покоя посчитать среднее значение показаний акселерометра
    for i in range(len(start)):
        sub_data = data[start[i]:end[i], :]
        total_x = np.sum(sub_data[:, ids[0]]) * scale
        total_y = np.sum(sub_data[:, ids[1]]) * scale
        total_z = np.sum(sub_data[:, ids[2]]) * scale
        counter = len(sub_data)

        avg_x = total_x / counter
        avg_y = total_y / counter
        avg_z = total_z / counter
        avg_list[0].append(avg_x)
        avg_list[1].append(avg_y)
        avg_list[2].append(avg_z)

    # выберем минимальное и максимальное среднее значение по каждой оси
    # это будут значения в момент воздействия гравитации на эту ось в обоих положениях
    for i in range(3):
        min_val[i] = np.min(avg_list[i])
        max_val[i] = np.max(avg_list[i])

    # для каждого оси посчитаем смещение и чувствительность
    bias = [0.0, 0.0, 0.0]
    sens = [0.0, 0.0, 0.0]
    for i in range(3):
        bias[i] = (min_val[i] + max_val[i]) / 2.0
        if np.sign(min_val[i]) == np.sign(max_val[i]):
            sens[i] = gravity / (abs(min_val[i] - max_val[i]) / 2.0)
        else:
            sens[i] = gravity / ((abs(min_val[i]) + abs(max_val[i])) / 2.0)

    params = {'scale_factor': scale,
              'bias': {'x': bias[0], 'y': bias[1], 'z': bias[2]},
              'sensitivity': {'x': sens[0], 'y': sens[1], 'z': sens[2]},
              'min': {'x': min_val[0], 'y': min_val[1], 'z': min_val[2]},
              'max': {'x': max_val[0], 'y': max_val[1], 'z': max_val[2]}}
    return params


def apply_calibration_parameters(data, accel_columns, accel_params, gyro_columns, gyro_params):
    new_data = np.zeros(data.shape)
    ac = accel_columns
    gc = gyro_columns
    # калибровать показания акселерометра
    a_scale = accel_params["scale_factor"]
    a_bias = accel_params["bias"]
    a_sens = accel_params["sensitivity"]
    new_data[:, ac[0]] = ((data[:, ac[0]] * a_scale) - a_bias["x"]) * a_sens["x"]
    new_data[:, ac[1]] = ((data[:, ac[1]] * a_scale) - a_bias["y"]) * a_sens["y"]
    new_data[:, ac[2]] = ((data[:, ac[2]] * a_scale) - a_bias["z"]) * a_sens["z"]
    # калибровать показания гироскопа
    g_scale = gyro_params["scale_factor"]
    g_bias = gyro_params["bias"]
    g_sens = gyro_params["sensitivity"]
    new_data[:, gc[0]] = ((data[:, gc[0]] * g_scale) - g_bias["x"]) * g_sens["x"]
    new_data[:, gc[1]] = ((data[:, gc[1]] * g_scale) - g_bias["y"]) * g_sens["y"]
    new_data[:, gc[2]] = ((data[:, gc[2]] * g_scale) - g_bias["z"]) * g_sens["z"]
    return new_data
