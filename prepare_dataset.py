import numpy as np
import os
import pandas as pd


# NOTE:
# change the paths at the bottom of this file!


def decompress_dataset(source_dir, target_dir):
    # Decompress NPZ into NPY files
    # NPY files are MUCH faster to load than NPZ files (but they take up more space)

    # source_dir has to be a path like: 'F:\\radar_data'
    # this directory should contain dirs for each year, which contain the npz files

    # this process takes a while, but it only has to be done once
    # it can only by sped up by parallelizing, if you have A LOT of RAM

    # NOTE: this process DOES NOT check if the files are already processed
    # if you run this process multiple times, undfined behaviour may occur

    # Create the target directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Decompress the dataset
    print('Decompressing dataset - This takes a while...')
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.npz') and '2024' in file:

                # load the npz file
                npz_file = os.path.join(root, file)
                data = np.load(npz_file, allow_pickle=True)

                # create the target directories if they do not exist
                npy_file = os.path.join(target_dir, os.path.relpath(npz_file, source_dir)[:-4] + '.npy')
                if not os.path.exists(os.path.dirname(npy_file)):
                    os.makedirs(os.path.dirname(npy_file))

                # file name inside of the NPZ archive changes between years ('arr_0', 'data', ...)
                assert len(list(data.files)) == 1, f'The npz file should contain only one array, but it contains {len(list(data.files))} arrays: {data.files}\n{npz_file}'

                # save the data as a npy file
                np.save(npy_file, data[data.files[0]])

                # delete data from memory
                del data
                print('Decompressed', npz_file, 'into', npy_file)


def decrease_resolution_of_specific_year(dir):
    # since 2023 is the only year wich has a resolution of 10 minutes
    # instead of 1 hour, we decrease the resolution of 2023 to 1 min
    # UPDATE: every year from 2023 onwards has a resolution of 10 minutes

    # dir has to be a path like: 'F:\\radar_data\\2023'

    # NOTE: this process checks if the files are already processed
    # and skips them if they are already processed

    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith('time.npy'):

                time_file = os.path.join(root, file)
                print(f'Processing {time_file}...', end='')

                # check if the file is already processed
                tmp = time_file.replace('time_full_res', 'time').replace('time', 'time_full_res')
                if os.path.exists(tmp):
                    print('\talready processed')
                    continue

                # load time file
                time_data = np.load(time_file, allow_pickle=True)
                
                # create boolean mask: True if minutes of timestamp is 50, False otherwise
                # seems like a weird rule, but all other files ar at XX:50
                mask = np.array([pd.Timestamp(x).minute == 50 for x in time_data], dtype=bool)
                
                # load data file
                data_file = time_file.replace('time', 'rw_values')
                data = np.load(data_file, allow_pickle=True)

                # aplly mask to time and data
                time_data = time_data[mask]
                data = data[mask]

                # rename original files
                os.rename(time_file, time_file.replace('time', 'time_full_res'))
                os.rename(data_file, data_file.replace('rw_values', 'rw_values_full_res'))

                # save new files
                np.save(time_file, time_data)
                np.save(data_file, data)

                # delete data from memory
                del time_data, data
                print('done')


if __name__ == '__main__':
    decompress_dataset(source_dir='compressed_data\\', target_dir='F:\\radar_data')
    decrease_resolution_of_specific_year(dir='F:\\radar_data\\2023')
    decrease_resolution_of_specific_year(dir='F:\\radar_data\\2024')
    print('Done!')
