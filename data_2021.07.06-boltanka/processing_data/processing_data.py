import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from numpy import genfromtxt
from datetime import datetime


def centering_data(data):
    for i in range(data.shape[1]):
        # нормировка
        min_val = data[:, i].min()
        max_val = data[:, i].max()
        diff = max_val - min_val
        data[:, i] /= diff / 2  # [-1;1]
        # смещение в ноль
        mean_val = data[:, i].mean()
        data[:, i] -= mean_val
        if (data[:, i].max()-1)>0:
           data[:, i] -= (data[:, i].max()-1)


def filtering_data(data, columns, kernel, times):
    from scipy import signal
    from scipy import interpolate
    for i in columns:
        # data[:, i] = interpolate.UnivariateSpline(times, data[:, i], k=2, s=0.2)
        data[:, i] = signal.savgol_filter(data[:, i], kernel, 1)
        # data[:, i] = signal.medfilt(data[:, i], kernel)


def read_sensor_data(path, imu):
    file_names = [f for f in listdir(path) if isfile(join(path, f))]

    imu_files = []
    for file_name in file_names:
        arr = file_name.split("-")
        if int(arr[1]) == imu:
            imu_files.append((int(arr[2]), file_name))
    imu_files.sort(key=lambda x: x[0])

    times = []
    accel_x = []
    accel_y = []
    accel_z = []
    gyro_x = []
    gyro_y = []
    gyro_z = []
    for num, file_name in imu_files:
        df = pd.read_csv(path + file_name, header=0, skiprows=0, delimiter=';')
        for index, row in df.iterrows():
            date_time_obj = datetime.strptime(row['server_time'], '%Y-%m-%dT%H:%M:%S.%f')
            accel = row['(ax,ay,az)'].replace('(', '').replace(')', '').split(',')
            ax = float(accel[0])
            ay = float(accel[1])
            az = float(accel[2])
            gyro = row['(gx,gy,gz)'].replace('(', '').replace(')', '').split(',')
            gx = float(gyro[0])
            gy = float(gyro[1])
            gz = float(gyro[2])

            times.append(date_time_obj)
            accel_x.append(ax)
            accel_y.append(ay)
            accel_z.append(az)
            gyro_x.append(gx)
            gyro_y.append(gy)
            gyro_z.append(gz)

    data = np.zeros((len(times), 6), dtype=float)
    data[:, 0] = accel_x
    data[:, 1] = accel_y
    data[:, 2] = accel_z
    data[:, 3] = gyro_x
    data[:, 4] = gyro_y
    data[:, 5] = gyro_z
    return times, data


def read_control_data(control_file):
    # Упрвление
    times = []
    angle_x = []
    angle_y = []
    angle_z = []
    gf_list = []
    df_control = pd.read_csv(control_file, header=0, skiprows=0, delimiter=';')
    for index, row in df_control.iterrows():
        date_time_obj = datetime.strptime(row['Timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
        x = float(row['X'])
        y = float(row['Y'])
        z = float(row['Z'])
        gf = float(row['GF'])
        times.append(date_time_obj)
        angle_x.append(x)
        angle_y.append(y)
        angle_z.append(z)
        gf_list.append(gf)

    data = np.zeros((len(times), 4), dtype=float)
    data[:, 0] = angle_x
    data[:, 1] = angle_y
    data[:, 2] = angle_z
    data[:, 3] = gf_list
    return times, data


def match_sensor_and_control_data(times_sensor, times_control, data_sensor, data_control):
    times = []

    itr = 0
    start = times_control[0]
    end = times_control[len(times_control) - 1]
    for i, t in enumerate(times_sensor):
        if start <= t <= end:
            times.append(t)

    data = np.zeros((len(times), 10), dtype=float)
    j = 0
    for i, t in enumerate(times_sensor):
        if start <= t <= end:
            data[j, :6] = data_sensor[i, :]
            if t > times_control[itr]:
                itr += 1
            data[j, 6:] = data_control[itr, :]
            j += 1
    # data: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, angle_x, angle_y, angle_z, gf
    return times, data


def show_data(times, data):
    ax = plt.axes()
    ax.plot(times, -data[:, 0], label='accel_x')
    ax.plot(times, data[:, 1], label='accel_y')
    ax.plot(times, data[:, 2], label='accel_z')
    ax.plot(times, data[:, 3], label='gyro_x')
    ax.plot(times, data[:, 4], label='gyro_y')
    ax.plot(times, data[:, 5], label='gyro_z')
    ax.plot(times, data[:, 6], label='angle_x')
    ax.plot(times, data[:, 7], label='angle_x')
    ax.plot(times, data[:, 8], label='angle_x')
    ax.plot(times, data[:, 9], label='gf')
    # ax.plot(times, centering_data(signal.medfilt(accel_mod, 151)), label='accel_mod')
    plt.title("Графики ЦФ-18")
    plt.legend(framealpha=1, frameon=True)
    plt.show()


def show_accel_data(path, control_file, imu):
    times_sensor, data_sensor = read_sensor_data(path, imu)
    times_control, data_control = read_control_data(path + control_file)
    times, data = match_sensor_and_control_data(times_sensor, times_control, data_sensor, data_control)

    accel_mod = []
    for i in range(len(data)):
        mod_val = math.sqrt(data[i, 0]**2 + data[i, 1]**2 + data[i, 2]**2)
        accel_mod.append(-mod_val)

    times = times[1000:]
    data = data[1000:, :]

    centering_data(data)
    filtering_data(data, columns=[0, 1, 2, 3, 4, 5], kernel=151, times=times)
    show_data(times, data)


def format_data(path, out_name):
    # imu,server_time,arduino_time,ax,ay,az,gx,gy,gz,mx,my,mz
    for imu in range(3):
        times_sensor, data_sensor = read_sensor_data(path, imu)
        times_msc = np.zeros((len(times_sensor)), dtype=int)
        for i in range(len(times_sensor)):
            dt = times_sensor[i]-times_sensor[0]
            times_msc[i] = int(dt.total_seconds() * 1000)
            if i > 1 and times_msc[i] < times_msc[i-1]:
                a = 1

        data_count = len(times_sensor)
        data = {'imu': [imu] * data_count,
                'server_time': times_sensor,
                'arduino_time': times_msc,
                'ax': data_sensor[:, 0].astype(int),
                'ay': data_sensor[:, 1].astype(int),
                'az': data_sensor[:, 2].astype(int),
                'gx': data_sensor[:, 3].astype(int),
                'gy': data_sensor[:, 4].astype(int),
                'gz': data_sensor[:, 5].astype(int),
                'mx': [0] * data_count,
                'my': [0] * data_count,
                'mz': [0] * data_count}
        df = pd.DataFrame.from_dict(data)
        df.to_csv(f"cf-{imu}-{1}-{out_name}", index=False)


def main():
    path = "../raw_cf18_data-06.07.2021.12.06.42 - boltanka1/"
    control_file = "cf_control_data-2021-07-06T12.22.06.784356.csv"
    format_data(path, "boltanka1-06.07.2021.12.06.42.csv")

    path = "../raw_cf18_data-06.07.2021.12.24.42 - boltanka2/"
    control_file = "cf_control_data-2021-07-06T12.32.32.217538.csv"
    format_data(path, "boltanka2-06.07.2021.12.24.42.csv")

    # show_accel_data(path, control_file, 0)
    # show_accel_data(path, control_file, 1)
    # show_accel_data(path, control_file, 2)


if __name__ == '__main__':
    main()
