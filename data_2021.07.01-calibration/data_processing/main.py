import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from os import listdir
from os.path import isfile, join
from numpy import genfromtxt


start = [6235, 10585, 13910, 16990, 20295, 23420, 26860, 30050, 33460, 36620]
end = [9330, 12470, 15805, 19030, 22145, 25330, 28770, 32190, 35350, 39140]
path = "../raw_calib_data/"


def get_data_from_files(_path, imu):
    file_names = [f for f in listdir(_path) if isfile(join(_path, f))]

    files = []
    for file_name in file_names:
        arr = file_name.split("-")
        if int(arr[1]) == imu:
            files.append((int(arr[2]), file_name))

    files.sort(key=lambda x: x[0])

    all_data = None
    for num, file_name in files:
        dd = genfromtxt(_path + file_name, delimiter=',', skip_header=1)
        if all_data is None:
            all_data = dd
        else:
            all_data = np.concatenate((all_data, dd), axis=0)
    return all_data


def show_accel_data(imu, sensor):
    title = sensor["title"]
    ids = sensor["ids"]

    all_data = get_data_from_files(path, imu)

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
    sub_titles = {-1: "module", 0: ", axis: X", 1: ", axis: Y", 2: ", axis: Z"}
    title = sensor["title"] + sub_titles[axis]
    ids = sensor["ids"]

    data = get_data_from_files(path, imu)
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
        if axis < 0:
            d = np.sqrt(selected_data[:, ids[0]] ** 2 + selected_data[:, ids[1]] ** 2 + selected_data[:, ids[2]] ** 2)

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
    sensor = accel
    show_overlap_data_imu(0, sensor, axis)
    show_overlap_data_imu(1, sensor, axis)
    show_overlap_data_imu(2, sensor, axis)


def show_data_imu(imu, sensor, axis):
    sub_titles = {0: ", axis: X", 1: ", axis: Y", 2: ", axis: Z"}
    title = sensor["title"] + sub_titles[axis]
    ids = sensor["ids"]

    data = get_data_from_files(path, imu)
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


def cut_stable_period(data):
    intervals = get_intervals()
    intervals = intervals[intervals[:, 1].argsort()]
    l = []
    for i in range(len(intervals)):
        l.append(data[intervals[i, 1]:intervals[i, 2]])
    return np.concatenate(l)


def get_synthetic_acceleration(gyr_grad, radius):
    gyr_rad = gyr_grad / 57.2958
    gyr_rad_2 = gyr_rad * gyr_rad
    acc_synt = radius * gyr_rad_2
    acc_one = np.full(acc_synt.shape, 9.81)
    return np.sqrt(acc_synt ** 2 + acc_one ** 2)


def show_calibrated_data():
    imu = 0
    df = pd.read_csv(f"../calibration_different_acceleration/imu{imu}-calibration_different_acceleration.csv")
    acc = np.asarray(np.sqrt(df['ax'] ** 2 + df['ay'] ** 2 + df['az'] ** 2))
    gyr = np.asarray(np.sqrt(df['gx'] ** 2 + df['gy'] ** 2 + df['gz'] ** 2))
    times = (df['time'] - df['time'][0]) / 1000

    mode = 4

    if mode == 0:
        ax = plt.axes()
        # ax.plot(times, acc, label="Accelerometer")
        ax.plot(times, df['ax'], label="ax")
        ax.plot(times, df['ay'], label="ay")
        ax.plot(times, df['az'], label="az")
        # ax.plot([*range(len(gyr))], gyr, label="Gyroscope")
        plt.title("Различные уровни перегрузки")
        plt.xlabel("sec")
        plt.ylabel("[m/s^2]")
        plt.legend(framealpha=1, frameon=True)
        plt.show()
    elif mode == 1:
        ax = plt.axes()
        # ax.plot(times, gyr, label="Gyroscope")
        ax.plot(times, df['gx'], label="gx")
        ax.plot(times, df['gy'], label="gy")
        ax.plot(times, df['gz'], label="gz")
        plt.title("Различные уровни перегрузки")
        plt.xlabel("sec")
        plt.ylabel("[grad/s]")
        plt.legend(framealpha=1, frameon=True)
        plt.show()
    elif mode == 2:
        acc1 = cut_stable_period(acc)
        gyr1 = cut_stable_period(gyr)

        r = 15.8
        acc_synt2 = get_synthetic_acceleration(gyr1, r)

        ax = plt.axes()
        ax.plot([*range(len(acc1))], acc1, label="Акселерометр")
        # ax.plot([*range(len(gyr1))], gyr1, label="ДУС")
        # ax.plot([*range(len(acc_synt2))], acc_synt2, label="Радиус: " + str(r))
        plt.xlabel("measurements")
        plt.ylabel("[m/s^2]")
        plt.title("Акселерометр")
        # plt.legend(framealpha=1, frameon=True)
        plt.show()
    elif mode == 3:
        gyr1 = cut_stable_period(gyr)
        ax = plt.axes()
        ax.plot([*range(len(gyr1))], gyr1, label="ДУС")
        plt.xlabel("measurements")
        plt.ylabel("[grad/s]")
        plt.title("ДУС")
        plt.show()
    elif mode == 4:
        acc1 = cut_stable_period(acc)
        gyr1 = cut_stable_period(gyr)
        # gyr1 = gyr1 * 0.921405
        # gyr1 = gyr1 * 0.973659
        # r = 15.8
        r = 18
        acc_synt = get_synthetic_acceleration(gyr1, r)

        ax = plt.axes()
        ax.plot([*range(len(acc1))], acc1, label="Измеренное")
        # ax.plot([*range(len(gyr1))], gyr1, label="ДУС")
        ax.plot([*range(len(acc_synt))], acc_synt, label="Расчетное")
        plt.xlabel("measurements")
        plt.ylabel("[m/s^2]")
        plt.title(f"Сравнение расчетного ускорения с фактическим, радиус: {r} метра. Датчик №{imu}")
        plt.legend(framealpha=1, frameon=True)
        plt.show()


def show_calibrating_data_1g_raw():
    def extract_calibration_data(_data):
        start = [6235, 10585, 13910, 16990, 20295, 23420, 26860, 30050, 33460, 36620]
        end = [9330, 12470, 15805, 19030, 22145, 25330, 28770, 32190, 35350, 39140]

        data_calibration = None
        # times_calibration = None
        for i in range(len(start)):
            sub_data = _data[start[i]:end[i], :]
            # sub_time = _times[start[i]:end[i]]
            if data_calibration is None:
                data_calibration = sub_data
                # times_calibration = sub_time
            else:
                data_calibration = np.concatenate((data_calibration, sub_data), axis=0)
                # times_calibration = np.concatenate((times_calibration, sub_time), axis=0)
        return data_calibration  # , times_calibration

    calibration_data = "../../data_2021.07.08-boltanka/raw_cf18_data-08.07.2021.12.10.26-boltanka_1_high+calibration/"
    data = get_data_from_files(calibration_data, 0)
    calib_data = extract_calibration_data(data)

    acc = np.asarray(np.sqrt(calib_data[:, 3] ** 2 + calib_data[:, 4] ** 2 + calib_data[:, 5] ** 2))
    ax = plt.axes()
    ax.plot([*range(len(acc))], acc, label="Accelerometer")
    ax.plot([*range(len(acc))], calib_data[:, 3], label="ax")
    ax.plot([*range(len(acc))], calib_data[:, 4], label="ay")
    ax.plot([*range(len(acc))], calib_data[:, 5], label="az")
    plt.legend(framealpha=1, frameon=True)
    plt.show()


def show_calibrating_data_1g():
    # file = "../boltanka-08.07.2021-high-amplitude-calibration/imu0-boltanka-08.07.2021-high-amplitude-calibration.csv.csv"
    file = "../boltanka-08.07.2021-high-amplitude-calibration_blank/imu0-boltanka-08.07.2021-high-amplitude-calibration.csv.csv"
    df = pd.read_csv(file)
    acc = np.asarray(np.sqrt(df['ax'] ** 2 + df['ay'] ** 2 + df['az'] ** 2))
    gyr = np.asarray(np.sqrt(df['gx'] ** 2 + df['gy'] ** 2 + df['gz'] ** 2))
    ax = plt.axes()
    ax.plot([*range(len(acc))], acc, label="Accelerometer")
    ax.plot([*range(len(acc))], df['ax'], label="ax")
    ax.plot([*range(len(acc))], df['ay'], label="ay")
    ax.plot([*range(len(acc))], df['az'], label="az")
    # ax.plot([*range(len(gyr))], gyr, label="Gyroscope")
    # plt.title("Зависимость калибровочных коэффициентов")
    plt.legend(framealpha=1, frameon=True)
    plt.show()


def get_time_scale(df):
    times = []
    for index, row in df.iterrows():
        date_time_obj = datetime.strptime(row['server_time'], '%Y-%m-%dT%H:%M:%S.%f')
        times.append(date_time_obj)
    return times


def cut_stable_period_g1(data):
    l = []
    for i in range(len(start)):
        l.append(data[start[i]:end[i]])
    return np.concatenate(l)


def show_comparison_data_1g():
    file_calib = "../boltanka-08.07.2021-high-amplitude-calibration/imu0-boltanka-08.07.2021-high-amplitude-calibration.csv.csv"
    file_no_calib = "../boltanka-08.07.2021-high-amplitude-calibration_blank/imu0-boltanka-08.07.2021-high-amplitude-calibration.csv.csv"
    df_calib = pd.read_csv(file_calib)
    df_no_calib = pd.read_csv(file_no_calib)

    acc_calib = np.asarray(np.sqrt(df_calib['ax'] ** 2 + df_calib['ay'] ** 2 + df_calib['az'] ** 2))
    acc_no_calib = np.asarray(np.sqrt(df_no_calib['ax'] ** 2 + df_no_calib['ay'] ** 2 + df_no_calib['az'] ** 2))
    gyr_calib = np.asarray(np.sqrt(df_calib['gx'] ** 2 + df_calib['gy'] ** 2 + df_calib['gz'] ** 2))
    gyr_no_calib = np.asarray(np.sqrt(df_no_calib['gx'] ** 2 + df_no_calib['gy'] ** 2 + df_no_calib['gz'] ** 2))
    times = get_time_scale(df_calib)

    mode = 1

    if mode == 0:
        ax = plt.axes()
        ax.plot(times, acc_calib, label="Calibrated Accelerometer")
        ax.plot(times, acc_no_calib, label="Not Calibrated Accelerometer")
        plt.xlabel("time")
        plt.ylabel("[m/s^2]")
        plt.title("Откалиброванные и не откалиброванные показания акселерометра")
        plt.legend(framealpha=1, frameon=True)
        plt.show()
    elif mode == 1:
        ax = plt.axes()
        ax.plot(times, gyr_calib, label="Calibrated Gyroscope")
        ax.plot(times, gyr_no_calib, label="Not Calibrated Gyroscope")
        plt.xlabel("time")
        plt.ylabel("[grad/s]")
        plt.title("Откалиброванные и не откалиброванные показания ДУС")
        plt.legend(framealpha=1, frameon=True)
        plt.show()
    elif mode == 2:
        acc1 = cut_stable_period_g1(acc_calib)
        acc2 = cut_stable_period_g1(acc_no_calib)
    elif mode == 3:
        gyr1 = cut_stable_period_g1(gyr_calib)
        gyr2 = cut_stable_period_g1(gyr_no_calib)
        ax = plt.axes()
        ax.plot([*range(len(gyr1))], gyr1, label="Calibrated Gyroscope")
        ax.plot([*range(len(gyr2))], gyr2, label="Not Calibrated Gyroscope")
        plt.xlabel("measurements")
        plt.ylabel("[grad/s]")
        plt.title("Откалиброванные и не откалиброванные показания ДУС")
        plt.legend(framealpha=1, frameon=True)
        plt.show()


def calculate_rotation():
    file_calib = "../boltanka-08.07.2021-high-amplitude-calibration/imu0-boltanka-08.07.2021-high-amplitude-calibration.csv.csv"
    df_calib = pd.read_csv(file_calib)
    gyr_calib = np.asarray(np.sqrt(df_calib['gx'] ** 2 + df_calib['gy'] ** 2 + df_calib['gz'] ** 2))

    times = get_time_scale(df_calib)
    # arduino_time = df_calib['arduino_time'][8818:11003].to_numpy()
    # dyr_modul = gyr_calib[8818:11003]
    arduino_time = df_calib['arduino_time'][8700:41300].to_numpy()
    dyr_modul = gyr_calib[877:41300]

    rotation = 0
    for i in range(len(arduino_time)-1):
        dt = (arduino_time[i+1] - arduino_time[i]) / 1000
        angle_velocity = dyr_modul[i]
        rotation += dt * angle_velocity

    print(f"Rotation: {rotation}")

    ax = plt.axes()
    ax.plot([*range(len(gyr_calib))], gyr_calib, label="Calibrated Gyroscope")
    # ax.plot(arduino_time, dyr_modul, label="Calibrated Gyroscope")

    plt.xlabel("time")
    plt.ylabel("[grad/s]")
    plt.title("Откалиброванные и не откалиброванные показания ДУС")
    plt.legend(framealpha=1, frameon=True)
    plt.show()


def main():
    # show_full_set_of_data()
    # show_overlap_data(-1)
    # show_separate_accel_data(2)
    show_calibrated_data()
    # show_calibrating_data_1g_raw()
    # show_calibrating_data_1g()
    # show_comparison_data_1g()
    # calculate_rotation()


if __name__ == '__main__':
    main()
