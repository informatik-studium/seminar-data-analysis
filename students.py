import numpy as np
from pathlib import Path
from datetime import datetime
import logging as log
import pyproj
from pyproj import CRS
from tqdm import tqdm
import json
from pprint import pprint
from matplotlib import pyplot as plt
import imageio
import os


RADOLAN_WKT = """PROJCS["Radolan projection",
GEOGCS["Radolan Coordinate System",
    DATUM["Radolan Kugel",
        SPHEROID["Erdkugel", 6370040.0, 0.0]
    ],
    PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG","8901"]],
    UNIT["degree", 0.017453292519943295],
    AXIS["Longitude", EAST],
    AXIS["Latitude", NORTH]
],
PROJECTION["Stereographic_North_Pole"],
PARAMETER["central_meridian", 10.0],
PARAMETER["Standard_Parallel_1", 60.0],
PARAMETER["scale_factor", 1.0],
PARAMETER["false_easting", 0.0],
PARAMETER["false_northing", 0.0],
UNIT["km", 1000.0],
AXIS["X", EAST],
AXIS["Y", NORTH],
AUTHORITY["EPSG","1000001"]
]
"""


def convert_radolan_to_wgs84(x: np.ndarray, y: np.ndarray):
    """
    Converts coordinates from the Radolan coordinate reference system (CRS) to the WGS84 CRS.

    Parameters:
    x (np.ndarray): Array of x-coordinates in the Radolan CRS.
    y (np.ndarray): Array of y-coordinates in the Radolan CRS.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Tuple containing the converted x-coordinates and y-coordinates in the WGS84 CRS.
    """

    radolan_crs = CRS.from_wkt(RADOLAN_WKT)
    wgs84_crs = CRS.from_epsg(4326)

    transformer = pyproj.Transformer.from_crs(radolan_crs, wgs84_crs, always_xy=True)

    return transformer.transform(x, y)


def get_wgs84_grid():
    """
    Returns a grid of WGS84 coordinates.

    Returns:
        wgs84_grid (numpy.ndarray): A grid of WGS84 coordinates with shape (900, 900, 2).
                                    The first dimension represents latitude, the second dimension represents longitude,
                                    and the third dimension represents the coordinates.
    """
    x_radolan_coords = np.arange(-522.9621669218559, 376.0378330781441+0.1, 1.0)
    y_radolan_coords = np.arange(-4658.144724265571,  -3759.1447242655713+0.1, 1.0)

    wgs84_coords = convert_radolan_to_wgs84(x_radolan_coords, y_radolan_coords)
    wgs84_coords = np.array(wgs84_coords).T
    wgs84_coords = np.flip(wgs84_coords, axis=1)

    lat = np.repeat(wgs84_coords[:, 0], 900).reshape(900, 900)
    lon = np.tile(wgs84_coords[:, 1], 900).T.reshape(900, 900)

    wgs84_grid = np.stack([lat, lon], axis=2)
    return wgs84_grid


def read_radar_data(path: Path, start_date: datetime, end_date: datetime) -> tuple[np.ndarray, np.ndarray]:
    start_year = start_date.year
    end_year = end_date.year

    path_list = []
    for year in range(start_year, end_year+1):
        path_list.append(path.joinpath(str(year)))
    
    files = []
    for path_year in path_list:
        # get all files in the directory
        all_files_year = list(path_year.glob("*.npz"))
        log.debug(f"Found {len(all_files_year)} files in {path_year}")
        all_files_year = [file for file in all_files_year if not str(file.name).endswith("time.npz")]
        files.extend(all_files_year)
    
    log.info(f"Found {len(files)} time files")
    if len(files) == 0:
        raise Exception(f"No files found for the date range {start_date} to {end_date}. You may need to download more data first.")

    start_date = np.datetime64(start_date)
    end_date = np.datetime64(end_date)

    # read radar data
    radar_data = []
    time_data = []
    for file in tqdm(files, total=len(files)):
        time_file = str(file).replace("_rw_values.npz", "_time.npz")
        time = np.load(time_file, allow_pickle=True)["arr_0"]
        bool_time_filter = (time >= start_date) & (time <= end_date)
        time = time[bool_time_filter]

        # skip file if no data is available
        if len(time) == 0:
            continue
        data = np.load(file, allow_pickle=True)["arr_0"]
        data = data[bool_time_filter]

        radar_data.append(data)
        time_data.append(time)
    
    np_time_data =  np.concatenate(time_data)
    np_radar_data = np.concatenate(radar_data)

    return np_radar_data, np_time_data


def get_german_border():
    path = Path("germany.json")
    with open(path, "r") as file:
        data = json.load(file)
    coordinates = data["features"][0]["geometry"]["coordinates"][0]
    coordinates = np.array(coordinates)[0] # (2374, 2)
    return coordinates[:, 0], coordinates[:, 1]


def create_gif():
    path = Path("data")
    start_date = datetime(2023, 12, 19)
    end_date = datetime(2023, 12, 20)

    # load radar data
    radar_data, time_data = read_radar_data(path, start_date, end_date)
   
    # project radar data to WGS84
    grid = get_wgs84_grid() # (900, 900, 2)
    radar_data_wgs84 = np.zeros((radar_data.shape[0], 900, 900))
    for i in range(radar_data.shape[0]):
        radar_data_wgs84[i] = radar_data[i]

    # load german border
    border_x, border_y = get_german_border() # (B,), (B,)

    # plot radar data
    plt.ion()
    for i in range(radar_data_wgs84.shape[0]):

        plt.clf()

        # set aspect ratio to be equal
        plt.gca().set_aspect('equal', adjustable='box')

        plt.imshow(radar_data_wgs84[i], extent=[grid[0, 0, 1], grid[0, -1, 1], grid[0, 0, 0], grid[-1, 0, 0]])
        plt.plot(border_x, border_y)
        plt.title(str(time_data[i])[0:16].replace("T", " "))
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")


        # save image to file
        plt.savefig(f"gif_frames/frame_{i:04d}.png")

        plt.pause(0.001)
    plt.ioff()
    plt.close()

    # create gif from images
    images = []
    for filename in os.listdir("gif_frames"):
        images.append(imageio.imread(f"gif_frames/{filename}"))
    imageio.mimsave('radar_data.gif', images)

    # remove images
    for filename in os.listdir("gif_frames"):
        os.remove(f"gif_frames/{filename}")


def print_uncompressed_filesize():

    total_size_gb = 0

    path_list = []
    path = Path("data")
    for year in range(2006, 2023+1):
        path_list.append(path.joinpath(str(year)))
    
    files = []
    for path_year in path_list:
        all_files_year = list(path_year.glob("*.npz"))
        all_files_year = [file for file in all_files_year if not str(file.name).endswith("time.npz")]
        files.extend(all_files_year)
    
    # read radar data
    radar_data = []
    time_data = []
    for file in tqdm(files, total=len(files)):
        time_file = str(file).replace("_rw_values.npz", "_time.npz")
        time = np.load(time_file, allow_pickle=True)["arr_0"]

        # skip file if no data is available
        if len(time) == 0:
            continue
        data = np.load(file, allow_pickle=True)["arr_0"]

        # compute size in GB
        size = data.size * data.itemsize
        size_gb = size / 1e9
        total_size_gb += size_gb

    print(f"Total size of uncompressed radar data: {total_size_gb:.2f} GB")


if __name__ == "__main__":

    #create_gif()

    print_uncompressed_filesize()
    
