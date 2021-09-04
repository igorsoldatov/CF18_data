import numpy as np
import matplotlib.pyplot as plt
import collections
import json
from os import walk
from datetime import datetime
from os import listdir
from os.path import isfile, join

import pandas as pd
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
        if len(parts) < 4:
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
        if len(parts) < 4:
            continue
        file_type = parts[0]
        imu_param = parts[1]
        pack_id = int(parts[2])
        if file_type == "cf" and imu_param == imu:
            file_names[pack_id] = file_name
    file_names_sorted = collections.OrderedDict(sorted(file_names.items()))
    return file_names_sorted


def get_full_data_from_files(path, imu_files):
    times = []
    data = None
    for num, file_name in enumerate(imu_files):
        df = pd.read_csv(path + file_name, header=0, skiprows=0, delimiter=',')
        data_tmp = genfromtxt(path + file_name, delimiter=',', skip_header=1)
        if data is None:
            data = data_tmp
        else:
            data = np.concatenate([data, data_tmp])
        for index, row in df.iterrows():
            date_time_obj = datetime.strptime(row['server_time'], '%Y-%m-%dT%H:%M:%S.%f')
            times.append(date_time_obj)

    return data, times


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


def calculate_calibration_parameters(path):
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


def read_and_concat_csv(path, imu):
    df_list = []
    file_names = get_files(path, imu)
    for pack_id, name in file_names.items():
        df = pd.read_csv(path + name, dtype={'ax': np.float64, 'ay': np.float64, 'az': np.float64,
                                             'gx': np.float64, 'gy': np.float64, 'gz': np.float64,
                                             'mx': np.float64, 'my': np.float64, 'mz': np.float64})
        df_list.append(df)
    data = pd.concat(df_list)
    return data


def apply_calibration(data, calibration_params):
    data.reset_index(drop=True, inplace=True)
    accel_params = calibration_params["accelerometer"]
    a_scale = accel_params["scale_factor"]
    a_bias = accel_params["bias"]
    a_sens = accel_params["sensitivity"]
    gyro_params = calibration_params["gyroscope"]
    g_scale = gyro_params["scale_factor"]
    g_bias = gyro_params["bias"]
    g_sens = gyro_params["sensitivity"]
    for index, row in data.iterrows():
        data.at[index, 'ax'] = ((row["ax"] * a_scale) - a_bias["x"]) * a_sens["x"]
        data.at[index, 'ay'] = ((row["ay"] * a_scale) - a_bias["y"]) * a_sens["y"]
        data.at[index, 'az'] = ((row["az"] * a_scale) - a_bias["z"]) * a_sens["z"]
        data.at[index, 'gx'] = ((row["gx"] * g_scale) - g_bias["x"]) * g_sens["x"]
        data.at[index, 'gy'] = ((row["gy"] * g_scale) - g_bias["y"]) * g_sens["y"]
        data.at[index, 'gz'] = ((row["gz"] * g_scale) - g_bias["z"]) * g_sens["z"]


def transform_data(path, start, end, out_file):
    with open('imu_calibration_parameters.json') as json_file:
        imu_calibration_params = json.load(json_file)
        sensors = get_available_sensors(path)
        for imu in sensors:
            sub_data = read_and_concat_csv(path, imu)
            sub_data = sub_data.iloc[start:end]
            apply_calibration(sub_data, imu_calibration_params[imu])
            sub_data.to_csv(f"imu{imu}-{out_file}", index=False)


def data_preview(path, start=0, end=0):
    sensors = get_available_sensors(path)
    for imu in sensors:
        file_names = get_files(path, imu)
        # data_raw, times = get_full_data_from_files(path, file_names.values())
        data_raw, time_sec_raw = get_data_from_files(path, file_names.values())
        if start > 0 and end > 0:
            data_raw = data_raw[start:end, :]
            times = times[start:end]
        a_ids = [3, 4, 5]
        accel = {"title": "Accelerometer", "ids": a_ids}
        show_data(data_raw, [*range(len(data_raw))], imu, accel)
        # show_data(data_raw, times, imu, accel)


def data_preview_calibrated(path, file, imu):
    data_raw, times = get_full_data_from_files(path, [file])
    a_ids = [3, 4, 5]
    accel = {"title": "Accelerometer", "ids": a_ids}
    show_data(data_raw, times, imu, accel)


def main():
    # calibration_data = "../data_08.07.2021-boltanka/raw_cf18_data-08.07.2021.12.10.26-boltanka_1_high+calibration/"
    # calculate_calibration_parameters(calibration_data)

    boltanka1 = "../data_08.07.2021-boltanka/raw_cf18_data-08.07.2021.12.10.26-boltanka_1_high+calibration/"
    # data_preview(boltanka1, start=50260, end=88600)
    transform_data(boltanka1, start=50260, end=88600, out_file="boltanka-08.07.2021-high-amplitude.csv")

    boltanka2 = "../data_08.07.2021-boltanka/raw_cf18_data-08.07.2021.12.34.48-boltanka_2_low/"
    # data_preview(boltanka2, start=7335, end=44100)
    transform_data(boltanka2, start=7335, end=44100, out_file="boltanka-08.07.2021-low-amplitude.csv")

    boltanka3 = "../data_06.07.2021-boltanka/raw_cf18_data_preformat-boltanka1/"
    # data_preview(boltanka3, start=23485, end=41200)
    transform_data(boltanka3, start=23485, end=41200, out_file="boltanka-06.07.2021-1.1.csv")
    # data_preview(boltanka3, start=61200, end=105050)
    transform_data(boltanka3, start=61200, end=105050, out_file="boltanka-06.07.2021-1.2.csv")

    boltanka5 = "../data_06.07.2021-boltanka/raw_cf18_data_preformat-boltanka2/"
    # data_preview(boltanka5, start=10200, end=51860)
    transform_data(boltanka5, start=10200, end=51860, out_file="boltanka-06.07.2021-2.1.csv")

    boltanka6 = "../data_02.07.2021-boltanka/raw_cf18_data_preformat-boltanka/"
    # data_preview(boltanka6, start=226400, end=242600)
    transform_data(boltanka6, start=226400, end=242600, out_file="boltanka-02.07.2021-1.1.csv")
    # data_preview(boltanka6, start=465000, end=485250)
    transform_data(boltanka6, start=465000, end=485250, out_file="boltanka-02.07.2021-1.2.csv")


if __name__ == '__main__':
    main()
    # data_preview_calibrated("../all_preprocessed_data/", "imu0-boltanka-08.07.2021-high-amplitude.csv", 0)
