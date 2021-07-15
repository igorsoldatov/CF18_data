import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from numpy import genfromtxt

path = "E:/Данные_с_центрифуги_ЦФ-18/data_08.07.2021-болтанка/raw_cf18_data-08.07.2021.11.39.47-fast-cabin-rotation/"


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

    time_sec = all_data[:, 2] / 1000

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(time_sec, all_data[:, ids[0]], label='x')
    ax.plot(time_sec, all_data[:, ids[1]], label='y')
    ax.plot(time_sec, all_data[:, ids[2]], label='z')
    plt.title(title)
    plt.legend(framealpha=1, frameon=True)
    plt.show()


def main():
    accel = {"title": "Accelerometer", "ids": [3, 4, 5]}
    gyro = {"title": "Gyroscope", "ids": [6, 7, 8]}
    magnet = {"title": "Magnetometer", "ids": [9, 10, 11]}
    current_sensor = accel
    show_accel_data(0, current_sensor)
    show_accel_data(1, current_sensor)
    show_accel_data(2, current_sensor)


if __name__ == '__main__':
    main()
