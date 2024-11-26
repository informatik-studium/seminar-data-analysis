import numpy as np
import os
import pandas as pd


def decompress_dataset(source_dir = 'compressed_data\\', target_dir = 'F:\\radar_data'):
    # Decompress NPZ into NPY files
    # NPY files are MUCH faster to load than NPZ files (but they take up more space)

    # Create the target directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Decompress the dataset
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.npz') and '2023' in file:

                # load the npz file
                npz_file = os.path.join(root, file)
                data = np.load(npz_file, allow_pickle=True)

                # create the target directories if they do not exist
                npy_file = os.path.join(target_dir, os.path.relpath(npz_file, source_dir)[:-4] + '.npy')
                if not os.path.exists(os.path.dirname(npy_file)):
                    os.makedirs(os.path.dirname(npy_file))

                # save the data as a npy file
                np.save(npy_file, data['arr_0'])

                # delete data from memory
                del data
                print('Decompressed', npz_file, 'into', npy_file)


def decrease_resolution_of_2023(dir = 'F:\\radar_data\\2023'):
    # since 2023 is the only year wich has a resolution of 10 minutes
    # instead of 1 hour, we decrease the resolution of 2023 to 1 min

    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith('time.npy'):

                time_file = os.path.join(root, file)
                print(f'Processing {time_file}...', end='')

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
    decompress_dataset()
    decrease_resolution_of_2023()
    print('Done!')
