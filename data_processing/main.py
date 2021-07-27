import numpy as np
import matplotlib.pyplot as plt
import collections
import json
from os import walk
from os import listdir
from os.path import isfile, join
from numpy import genfromtxt
from imu_calibration import *


# start = [4360, 6235, 10585, 13910, 16990, 20295, 23420, 26860, 30050, 33460, 36620]
# end = [5795, 9330, 12470, 15805, 19030, 22145, 25330, 28770, 32190, 35350, 39140]
start = [6235, 10585, 13910, 16990, 20295, 23420, 26860, 30050, 33460, 36620]
end = [9330, 12470, 15805, 19030, 22145, 25330, 28770, 32190, 35350, 39140]


def get_available_sensors(path):
    imus = []
    all_file_names = next(walk(path), (None, None, []))[2]
    for file_name in all_file_names:
        parts = file_name.split('-')
        if len(parts) != 4:
            continue
        file_type = parts[0]
        imu = parts[1]
        if file_type == "cf":
            if imu not in imus:
                imus.append(imu)
    return imus


def get_files(path, imu):
    file_names = {}
    all_file_names = next(walk(path), (None, None, []))[2]
    for file_name in all_file_names:
        parts = file_name.split('-')
        if len(parts) != 4:
            continue
        file_type = parts[0]
        imu_param = parts[1]
        pack_id = int(parts[2])
        if file_type == "cf" and imu_param == imu:
            file_names[pack_id] = file_name
    file_names_sorted = collections.OrderedDict(sorted(file_names.items()))
    return file_names_sorted


def get_data_from_files(path, imu_files):
    data = None
    for file_name in imu_files:
        dd = genfromtxt(path + file_name, delimiter=',', skip_header=1)
        if data is None:
            data = dd
        else:
            data = np.concatenate((data, dd), axis=0)

    time_sec = data[:, 2] / 1000
    return data, time_sec


def extract_calibration_data(data, times):
    data_calibration = None
    times_calibration = None
    for i in range(len(start)):
        sub_data = data[start[i]:end[i], :]
        sub_time = times[start[i]:end[i]]
        if data_calibration is None:
            data_calibration = sub_data
            times_calibration = sub_time
        else:
            data_calibration = np.concatenate((data_calibration, sub_data), axis=0)
            times_calibration = np.concatenate((times_calibration, sub_time), axis=0)
    return data_calibration, times_calibration


def show_data(data, times, imu, sensor):
    title = sensor["title"]
    ids = sensor["ids"]
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(times, data[:, ids[0]], label='x')
    ax.plot(times, data[:, ids[1]], label='y')
    ax.plot(times, data[:, ids[2]], label='z')
    plt.title(f"{title}, IMU: {imu}")
    plt.legend(framealpha=1, frameon=True)
    plt.show()


def calculate_calibration_parameters():
    path = "../../Данные_с_центрифуги_ЦФ-18/data_08.07.2021-boltanka/data/raw_cf18_data-08.07.2021.12.10.26-boltanka_1_high+calibration/"
    accel_range = AccelRange.ACCEL_RANGE_16G
    gyro_range = GyroRange.GYRO_RANGE_2000DPS

    a_ids = [3, 4, 5]
    accel = {"title": "Accelerometer", "ids": a_ids}
    g_ids = [6, 7, 8]
    gyro = {"title": "Gyroscope", "ids": g_ids}
    m_ids = [9, 10, 11]
    magnet = {"title": "Magnetometer", "ids": m_ids}
    show_setting = accel

    imu_calibration_params = {}
    sensors = get_available_sensors(path)
    for imu in sensors:
        file_names = get_files(path, imu)
        # все данные
        data_raw, time_sec_raw = get_data_from_files(path, file_names.values())
        # только данные процесса калибровки
        data, times = extract_calibration_data(data_raw, time_sec_raw)
        # show_data(data, times, imu, show_setting)

        accel_params = calibrate_accelerometer(data_raw, start, end, accel, accel_range)
        gyro_params = calibrate_gyroscope(data, gyro, gyro_range)
        imu_calibration_params[imu] = {"imu": int(imu),
                                       "reference": False,
                                       "accelerometer": accel_params,
                                       "gyroscope": gyro_params,
                                       "magnetometer": {}}

        data_calibrated = apply_calibration_parameters(data, a_ids, accel_params, g_ids, gyro_params)
        # show_data(data_calibrated, times, imu, show_setting)

    with open(f"imu_calibration_parameters.json", "w") as outfile:
        json.dump(imu_calibration_params, outfile)


def apply_calibration_parameters():
    imu_calibration_params = {}
    with open('imu_calibration_parameters.json') as json_file:
        imu_calibration_params = json.load(json_file)


def main():
    calculate_calibration_parameters()
    apply_calibration_parameters()


if __name__ == '__main__':
    main()
