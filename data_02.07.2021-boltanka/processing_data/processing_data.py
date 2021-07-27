import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import datetime

import pandas as pd
from numpy import genfromtxt

path = "../raw_calib_data-02.07.2021.11.24.28-boltanka/"


def show_accel_data(imu, sensor):
    title = sensor["title"]
    ids = sensor["ids"]
    file_names = [f for f in listdir(path) if isfile(join(path, f))]

    data = []
    for file_name in file_names:
        arr = file_name.split("-")
        if int(arr[1]) == imu:
            data.append((int(arr[2]), file_name))

    data.sort(key=lambda x: x[0])

    all_data = None
    for num, file_name in data:
        dd = genfromtxt(path + file_name, delimiter=',', skip_header=1)
        if all_data is None:
            all_data = dd
        else:
            all_data = np.concatenate((all_data, dd), axis=0)

    time_sec = all_data[:, 1] / 1000

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(time_sec, all_data[:, ids[0]], label='x')
    ax.plot(time_sec, all_data[:, ids[1]], label='y')
    ax.plot(time_sec, all_data[:, ids[2]], label='z')
    plt.title(title)
    plt.legend(framealpha=1, frameon=True)
    plt.show()


def format_data(imu):
    file_names = [f for f in listdir(path) if isfile(join(path, f))]

    imu_files = []
    for file_name in file_names:
        arr = file_name.split("-")
        if int(arr[1]) == imu:
            imu_files.append((int(arr[2]), file_name))

    imu_files.sort(key=lambda x: x[0])

    df_list = []
    for num, file_name in imu_files:
        df = pd.read_csv(path + file_name)
        df_list.append(df)
    new_df = pd.concat(df_list)

    times = []
    for index, row in new_df.iterrows():
        minutes = int(row['time'] / (60000 * 60))
        sec = int((row['time'] % 60000) / 1000)
        msc = row['time'] % 1000
        times.append(datetime.datetime(year=2021, month=7, day=2, hour=11, minute=minutes, second=sec, microsecond=msc*1000))

    data = {'imu': new_df["imu"],
            'server_time': times,
            'arduino_time': new_df["time"],
            'ax': new_df["ax"],
            'ay': new_df["ay"],
            'az': new_df["az"],
            'gx': new_df["gx"],
            'gy': new_df["gy"],
            'gz': new_df["gz"],
            'mx': new_df["mx"],
            'my': new_df["my"],
            'mz': new_df["mz"]}
    df = pd.DataFrame.from_dict(data)
    out_name = "boltanka-02.07.2021.csv"
    df.to_csv(f"cf-{imu}-{1}-{out_name}", index=False)


def main():
    format_data(0)
    format_data(1)
    format_data(2)

    accel = {"title": "Accelerometer", "ids": [2, 3, 4]}
    gyro = {"title": "Gyroscope", "ids": [5, 6, 7]}
    magnet = {"title": "Magnetometer", "ids": [8, 9, 10]}
    current_sensor = accel
    show_accel_data(0, current_sensor)
    show_accel_data(1, current_sensor)
    show_accel_data(2, current_sensor)


if __name__ == '__main__':
    main()
