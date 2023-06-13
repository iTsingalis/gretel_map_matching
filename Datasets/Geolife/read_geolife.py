import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os


def read_plt(plt_file):
    points = pd.read_csv(plt_file, skiprows=6, header=None,
                         parse_dates=[[5, 6]], infer_datetime_format=True)
    traj_id = os.path.basename(plt_file)[:-4]
    # for clarity rename columns
    points.rename(inplace=True, columns={'5_6': 'time', 0: 'lat', 1: 'lon', 3: 'alt'})

    # remove unused columns
    points.drop(inplace=True, columns=[2, 4])

    points['traj_id'] = traj_id

    return points


mode_names = ['walk', 'bike', 'bus', 'car', 'subway', 'train', 'airplane', 'boat', 'run', 'motorcycle', 'taxi']
mode_ids = {s: i + 1 for i, s in enumerate(mode_names)}


def read_labels(labels_file):
    labels = pd.read_csv(labels_file, skiprows=1, header=None,
                         parse_dates=[[0, 1], [2, 3]],
                         infer_datetime_format=True, delim_whitespace=True)

    # for clarity rename columns
    labels.columns = ['start_time', 'end_time', 'label']

    # replace 'label' column with integer encoding
    labels['label'] = [mode_ids[i] for i in labels['label']]

    return labels


def apply_labels(points, labels):
    indices = labels['start_time'].searchsorted(points['time'], side='right') - 1
    no_label = (indices < 0) | (points['time'].values >= labels['end_time'].iloc[indices].values)
    points['label'] = labels['label'].iloc[indices].values
    points['label'][no_label] = 0


def read_user(user_folder):
    labels = None

    plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
    df = pd.concat([read_plt(f) for f in plt_files])

    labels_file = os.path.join(user_folder, 'labels.txt')
    if os.path.exists(labels_file):
        labels = read_labels(labels_file)
        apply_labels(df, labels)
    else:
        df['label'] = 0

    return df


def read_all_users_geolife(folder):
    subfolders = os.listdir(folder)
    dfs = []
    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing user %s' % (i + 1, len(subfolders), sf))
        df = read_user(os.path.join(folder, sf))
        df['user'] = int(sf)
        dfs.append(df)
    return pd.concat(dfs)


def read_geolife_gps(gps_file, latMin, latMax, longMin, longMax, min_len=100):
    df = pd.read_pickle(gps_file)
    # mode_names = ['no label', 'walk(0)', 'bike(1)', 'bus(2)', 'car(3)', 'subway(4)', 'train(5)',
    #               'airplane(6)', 'boat(7)', 'run(8)', 'motorcycle(9)', 'taxi(10)']
    # df = df[df['label'].isin([3, 2, 10])]

    # ax = df['label'].value_counts().plot(kind='bar')
    # ax.set_xticklabels(mode_names, rotation=45, ha='right')
    # ax.set_yscale('log')
    # ax.set_ylabel('log count of labels')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    df = df[df['label'].isin([3])]  # is car
    df = df.groupby('traj_id').filter(lambda x: len(x) > min_len)

    df['inside'] = (df['lon'] > longMin) & (df['lon'] < longMax) & (df['lat'] > latMin) & (df['lat'] < latMax)
    # df = df[mask]
    print('proportion of points inside the OSM box {:.2f}%'.format(df['inside'].mean() * 100))

    traj_inside = df.groupby('traj_id').all()['inside']
    keep_points = traj_inside[df.traj_id]
    df = df.loc[keep_points.values]

    return df
