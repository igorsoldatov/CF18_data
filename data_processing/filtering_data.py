import os
from os import walk
import pandas
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import numpy as np


def apply_filter(data):
    b, a = butter(3, 0.02)
    # ax, ay, az, gx, gy, gz, mx, my, mz
    data['ax'] = filtfilt(b, a, data['ax'])
    data['ay'] = filtfilt(b, a, data['ay'])
    data['az'] = filtfilt(b, a, data['az'])
    data['gx'] = filtfilt(b, a, data['gx'])
    data['gy'] = filtfilt(b, a, data['gy'])
    data['gz'] = filtfilt(b, a, data['gz'])
    data['mx'] = filtfilt(b, a, data['mx'])
    data['my'] = filtfilt(b, a, data['my'])
    data['mz'] = filtfilt(b, a, data['mz'])


def show_all_data(data, title):
    # scale canvas
    minmax = [data['ax'].min(), data['ax'].max(), data['ay'].min(), data['ay'].max(), data['az'].min(),
              data['az'].max()]
    min_len = min(minmax)
    max_len = max(minmax)
    plt.plot([min_len, max_len], [min_len, max_len], '.', color='white')

    plt.plot(data['ay'], data['az'], '.', color='black')
    plt.plot(data['ax'], data['az'], '.', color='red')
    plt.plot(data['ax'], data['ay'], '.', color='green')
    plt.title(title)
    plt.show()


def show_data(data, title, start=0, out=""):
    plt.clf()
    ax1 = [data['ay'], data['ax'], data['ax']]
    ax2 = [data['az'], data['az'], data['ay']]
    ax3 = [data['ax'], data['ay'], data['az']]
    axes1 = ['ay', 'ax', 'ax']
    axes2 = ['az', 'az', 'ay']
    labels = ['ay*az', 'ax*az', 'ax*ay']
    min_val = 99999999
    min_idx = -1
    for i in range(3):
        check_sum = abs(min(ax1[i])) + abs(min(ax2[i]))
        if check_sum < min_val:
            min_val = check_sum
            min_idx = i

    if min_idx >= 0:
        if False:
            val_range = [min(ax1[min_idx][start:]), max(ax1[min_idx][start:]),
                         min(ax2[min_idx][start:]), max(ax2[min_idx][start:])]
            min_len = min(val_range)
            max_len = max(val_range)
            print(f"min: {min_len}, max: {max_len}")
            plt.plot([min_len, max_len], [min_len, max_len], '.', color='white')
        if True:
            max1 = max(ax1[min_idx][start:])
            min1 = min(ax1[min_idx][start:])
            range1 = max1 - min1
            max2 = max(ax2[min_idx][start:])
            min2 = min(ax2[min_idx][start:])
            range2 = max2 - min2
            max_range = max(range1, range2)
            plt.plot([min1, min1+max_range], [min2, min2+max_range], '.', color='white')

        colorlist = ax3[min_idx][start:]

        plt.scatter(ax1[min_idx][start:], ax2[min_idx][start:], c=colorlist, marker='.', s=0.5,
                    label=labels[min_idx], cmap='viridis')
        plt.xlabel(f"{axes1[min_idx]} [m/s^2]")
        plt.ylabel(f"{axes2[min_idx]} [m/s^2]")
        plt.legend(framealpha=1, frameon=True)
        plt.colorbar()
        plt.title(title)
        if len(out) > 0:
            plt.savefig(out)
        else:
            plt.show()


def main():
    path_in = "../all_preprocessed_data/2021.07.02/"
    path_out = "../all_preprocessed_data/2021.07.02-filtered/"
    all_folders = next(walk(path_in), (None, None, []))[1]
    for folder_in in all_folders:
        folder_out = f"{path_out}{folder_in}/"
        if not os.path.exists(folder_out):
            os.makedirs(folder_out)

        all_files = next(walk(f"{path_in}{folder_in}/"), (None, None, []))[2]

        for file in all_files:
            df = pandas.read_csv(f"{path_in}{folder_in}/{file}")
            # show_data(df, file)
            apply_filter(df)
            # show_data(df, file, 10000)
            df.to_csv(f"{folder_out}/{file}", index=False)


def visualization():
    path = "../all_preprocessed_data"
    folders = ["2021.07.02-filtered", "2021.07.06-filtered", "2021.07.08-filtered", "2021.09.03-filtered"]
    for folder in folders:
        experiments = next(walk(f"{path}/{folder}/"), (None, None, []))[1]
        for exp in experiments:
            folder_out = f"{path}/{folder}-xy/{exp}"
            if not os.path.exists(folder_out):
                os.makedirs(folder_out)
            exp_folder = f"{path}/{folder}/{exp}/"
            all_files = next(walk(exp_folder), (None, None, []))[2]
            for file in all_files:
                df = pandas.read_csv(f"{exp_folder}/{file}")
                show_data(df, file, 10000, out=f"{folder_out}/{file}.png")


def field_uniformity_accelerations():
    path = "../all_preprocessed_data"
    # folders = ["2021.07.02-filtered", "2021.07.06-filtered", "2021.07.08-filtered", "2021.09.03-filtered"]
    folders = ["2021.07.08-filtered", "2021.09.03-filtered"]
    for folder in folders:
        experiments = next(walk(f"{path}/{folder}/"), (None, None, []))[1]
        for exp in experiments:
            folder_out = f"{path}/{folder}-xy/{exp}"
            if not os.path.exists(folder_out):
                os.makedirs(folder_out)
            exp_folder = f"{path}/{folder}/{exp}/"
            all_files = next(walk(exp_folder), (None, None, []))[2]

            ax = plt.axes()

            for imu, file in enumerate(all_files):
                df = pandas.read_csv(f"{exp_folder}/{file}")
                acc = np.asarray(np.sqrt(df['ax'] ** 2 + df['ay'] ** 2 + df['az'] ** 2))
                ax.plot([*range(len(acc))], acc, label="IMU #" + str(imu))

            plt.xlabel("measurements")
            plt.ylabel("[m/s^2]")
            plt.title("Однородность поля перегрузок в кабине ЦФ18. " + exp)
            plt.legend(framealpha=1, frameon=True)
            plt.show()


if __name__ == '__main__':
    # main()
    visualization()
    # field_uniformity_accelerations()

