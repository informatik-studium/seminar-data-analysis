import numpy as np
from pathlib import Path
from datetime import datetime
import pyproj
from pyproj import CRS
from tqdm import tqdm
import json
import time
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


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


def read_radar_data(start_date: datetime, end_date: datetime) -> tuple[np.ndarray, np.ndarray]:
    start_year = start_date.year
    end_year = end_date.year
    path = Path("F:\\radar_data")

    path_list = []
    for year in range(start_year, end_year+1):
        path_list.append(path.joinpath(str(year)))
    
    files = []
    for path_year in path_list:
        # get all files in the directory
        all_files_year = list(path_year.glob("*.npy"))
        all_files_year = [file for file in all_files_year if not str(file.name).endswith("time.npy") and not str(file.name).endswith("full_res.npy")]
        files.extend(all_files_year)
    
    if len(files) == 0:
        raise Exception(f"No files found for the date range {start_date} to {end_date}. You may need to download more data first.")
    
    start_date = np.datetime64(start_date)
    end_date = np.datetime64(end_date)

    # read radar data
    radar_data = []
    time_data = []

    for file in files:
        time_file = str(file).replace("_rw_values.npy", "_time.npy")
        time = np.load(time_file, allow_pickle=True)
        bool_time_filter = (time >= start_date) & (time <= end_date)
        time = time[bool_time_filter]

        # skip file if no data is available
        if len(time) == 0:
            continue
        data = np.load(file, allow_pickle=False)
        data = data[bool_time_filter]
        
        radar_data.append(data)
        time_data.append(time)
    
    np_time_data =  np.concatenate(time_data)
    np_radar_data = np.concatenate(radar_data)

    return np_radar_data, np_time_data


def print_uncompressed_filesize():

    total_size_gb = 0
    start_time = time.perf_counter()

    path_list = []
    path = Path("F:\\radar_data")
    for year in range(2006, 2023+1):
        path_list.append(path.joinpath(str(year)))
    
    files = []
    for path_year in path_list:
        all_files_year = list(path_year.glob("*.npy"))
        all_files_year = [file for file in all_files_year if not str(file.name).endswith("time.npy") and not str(file.name).endswith("full_res.npy")]
        files.extend(all_files_year)
    
    # read radar data
    for file in tqdm(files, total=len(files)):
        time_file = str(file).replace("_rw_values.npy", "_time.npy")
        time_ = np.load(time_file, allow_pickle=True)

        # skip file if no data is available
        if len(time_) == 0:
            continue
        data = np.load(file, allow_pickle=True)

        # compute size in GB
        size = data.size * data.itemsize
        size_gb = size / 1e9
        total_size_gb += size_gb

        # delete data
        del data

    print(f"Total size of uncompressed radar data: {total_size_gb:.2f} GB")
    print(f"Time taken: {time.perf_counter() - start_time:.2f} seconds")


def get_NRW_mask():

    # check if mask already exists
    if Path("borders/nrw_mask.npy").exists():
        return np.load("borders/nrw_mask.npy")

    print("Creating mask for North Rhine-Westphalia...")

    # mask out North Rhine-Westphalia
    data_mask = np.zeros((900, 900), dtype='bool')
    
    # read border file
    path = Path("borders/german_states.json")
    with open(path, mode="r", encoding="utf-8") as file:
        data = json.load(file)
    data = data["features"]
    for state_data in data:
        if state_data["properties"]["name"] != 'Nordrhein-Westfalen':
            continue
        state_polygon = state_data["geometry"]["coordinates"][0]
        x, y = np.array(state_polygon).T
        state_polygon = Polygon(np.stack([x, y], axis=1))
        break
    
    # use point in polygon method to create mask
    grid = get_wgs84_grid()
    for i in range(900):
        for j in range(900):
            point = Point(grid[i, j][::-1])
            data_mask[i, j] = not state_polygon.contains(point)

    # save to file
    np.save("borders/nrw_mask.npy", data_mask)

    return data_mask


def get_germany_mask():

    if Path("borders/germany_mask.npy").exists():
        return np.load("borders/germany_mask.npy")
    
    print("Creating mask for Germany...")

    # load border
    path = Path("borders/germany.json")
    with open(path, "r") as file:
        data = json.load(file)
    coordinates = data["features"][0]["geometry"]["coordinates"][0]
    coordinates = np.array(coordinates)[0]
    polygon = Polygon(coordinates)

    # create mask
    data_mask = np.zeros((900, 900), dtype='bool')
    grid = get_wgs84_grid()
    for i in range(900):
        for j in range(900):
            point = Point(grid[i, j][::-1])
            data_mask[i, j] = not polygon.contains(point)

    # save to file
    np.save("borders/germany_mask.npy", data_mask)

    return data_mask



if __name__ == '__main__':
    mask = get_NRW_mask()
    from matplotlib import pyplot as plt    
    plt.imshow(mask, origin='lower')
    plt.show()
