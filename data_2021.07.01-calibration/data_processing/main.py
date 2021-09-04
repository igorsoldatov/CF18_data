import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from numpy import genfromtxt

path = "../raw_calib_data/"


def get_data_from_files(imu):
    file_names = [f for f in listdir(path) if isfile(join(path, f))]

    files = []
    for file_name in file_names:
        arr = file_name.split("-")
        if int(arr[1]) == imu:
            files.append((int(arr[2]), file_name))

    files.sort(key=lambda x: x[0])

    all_data = None
    for num, file_name in files:
        dd = genfromtxt(path + file_name, delimiter=',', skip_header=1)
        if all_data is None:
            all_data = dd
        else:
            all_data = np.concatenate((all_data, dd), axis=0)
    return all_data


def show_accel_data(imu, sensor):
    title = sensor["title"]
    ids = sensor["ids"]

    all_data = get_data_from_files(imu)

    # time_sec = all_data[:, 1] / 1000
    indexes = [*range(len(all_data))]

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(indexes, all_data[:, ids[0]], label='x')
    ax.plot(indexes, all_data[:, ids[1]], label='y')
    ax.plot(indexes, all_data[:, ids[2]], label='z')
    plt.title(title)
    plt.legend(framealpha=1, frameon=True)
    plt.show()


def show_full_set_of_data():
    accel = {"title": "Accelerometer", "ids": [2, 3, 4]}
    gyro = {"title": "Gyroscope", "ids": [5, 6, 7]}
    magnet = {"title": "Magnetometer", "ids": [8, 9, 10]}
    current_sensor = gyro
    show_accel_data(0, current_sensor)
    show_accel_data(1, current_sensor)
    show_accel_data(2, current_sensor)


def get_intervals():
    data_intervals = genfromtxt("intervals.csv", delimiter=',', skip_header=1, dtype=int)
    return data_intervals


def show_overlap_data_imu(imu, sensor, axis):
    sub_titles = {0: ", axis: X", 1: ", axis: Y", 2: ", axis: Z"}
    title = sensor["title"] + sub_titles[axis]
    ids = sensor["ids"]

    data = get_data_from_files(imu)
    intervals = get_intervals()
    g_min = min(intervals[:, 0])
    g_max = max(intervals[:, 0])
    min_high = 0
    max_high = 0

    min_values = []
    max_values = []

    ax = plt.axes()

    for g in range(g_min, g_max+1):
        selected_data = None
        for i in range(len(intervals)):
            g_curr = intervals[i, 0]
            if g_curr != g:
                continue
            start = intervals[i, 1]
            # end = intervals[i, 2]
            end = start + 2360
            data_tmp = data[start:end, :]
            if selected_data is None:
                selected_data = data_tmp
            else:
                selected_data = np.concatenate([selected_data, data_tmp])

        label = 'g=' + str(round(1 + (0.2 * g), 2))

        indexes = [*range(len(selected_data))]
        d = selected_data[:, ids[axis]]
        if min(d) < min_high:
            min_high = min(d)
        if max(d) > max_high:
            max_high = max(d)
        min_values.append(min(d))
        max_values.append(max(d))
        # fig = plt.figure()
        ax.plot(indexes, d, label=label)
        # ax.plot(indexes, selected_data[:, ids[1]], label='y')
        # ax.plot(indexes, selected_data[:, ids[2]], label='z')

    separators = []
    for i in range(0, 7):
        separators.append(i * 2360)
    plt.vlines(x=separators, ymin=min_high, ymax=max_high, colors='purple', ls='--', lw=1,)

    plt.title(title)
    plt.legend(framealpha=1, frameon=True)
    plt.show()

    ax = plt.axes()
    ax.plot([*range(g_min, g_max + 1)], min_values, label="min range")
    ax.plot([*range(g_min, g_max + 1)], max_values, label="max range")
    plt.title("Зависимость калибровочных коэффициентов")
    plt.legend(framealpha=1, frameon=True)
    plt.show()


def show_overlap_data(axis):
    accel = {"title": "Accelerometer", "ids": [2, 3, 4]}
    gyro = {"title": "Gyroscope", "ids": [5, 6, 7]}
    magnet = {"title": "Magnetometer", "ids": [8, 9, 10]}
    sensor = gyro
    show_overlap_data_imu(0, sensor, axis)
    show_overlap_data_imu(1, sensor, axis)
    show_overlap_data_imu(2, sensor, axis)


def show_data_imu(imu, sensor, axis):
    sub_titles = {0: ", axis: X", 1: ", axis: Y", 2: ", axis: Z"}
    title = sensor["title"] + sub_titles[axis]
    ids = sensor["ids"]

    data = get_data_from_files(imu)
    intervals = get_intervals()
    g_min = min(intervals[:, 0])
    g_max = max(intervals[:, 0])

    min_values = []
    max_values = []

    for g in range(g_min, g_max+1):
        selected_data = None
        for i in range(len(intervals)):
            g_curr = intervals[i, 0]
            if g_curr != g:
                continue
            start = intervals[i, 1]
            # end = intervals[i, 2]
            end = start + 2360
            data_tmp = data[start:end, :]
            if selected_data is None:
                selected_data = data_tmp
            else:
                selected_data = np.concatenate([selected_data, data_tmp])

        label = 'g=' + str(round(1 + (0.2 * g), 2))

        indexes = [*range(len(selected_data))]
        d = selected_data[:, ids[axis]]
        ax = plt.axes()
        ax.plot(indexes, d, label=label)
        # ax.plot(indexes, selected_data[:, ids[1]], label='y')
        # ax.plot(indexes, selected_data[:, ids[2]], label='z')

        separators = []
        for i in range(0, 7):
            separators.append(i * 2360)
        min_high = min(d)
        max_high = max(d)
        min_values.append(min_high)
        max_values.append(max_high)
        plt.vlines(x=separators, ymin=min_high, ymax=max_high, colors='purple', ls='--', lw=1,)

        plt.title(title)
        plt.legend(framealpha=1, frameon=True)
        plt.show()

    ax = plt.axes()
    ax.plot([*range(g_min, g_max + 1)], min_values, label="min range")
    ax.plot([*range(g_min, g_max + 1)], max_values, label="max range")
    plt.title("Зависимость калибровочных коэффициентов")
    plt.legend(framealpha=1, frameon=True)
    plt.show()


def show_separate_accel_data(axis):
    accel = {"title": "Accelerometer", "ids": [2, 3, 4]}
    gyro = {"title": "Gyroscope", "ids": [5, 6, 7]}
    magnet = {"title": "Magnetometer", "ids": [8, 9, 10]}
    sensor = gyro
    show_data_imu(0, sensor, axis)
    show_data_imu(1, sensor, axis)
    show_data_imu(2, sensor, axis)


def main():
    # show_full_set_of_data()
    show_overlap_data(2)
    # show_separate_accel_data(2)


if __name__ == '__main__':
    main()
