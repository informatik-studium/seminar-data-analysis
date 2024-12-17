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
import matplotlib.ticker as mticker  
import imageio.v2 as imageio
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from utils import *
from plotting_utils import *


def create_gif():
    path = Path("F:\\radar_data")
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2019, 12, 31)

    print(f'[GIF] Creating gif from radar data between {start_date} and {end_date}')

    # create directory for gif frames
    if not os.path.exists("gif_frames"):
        os.makedirs("gif_frames")
    else:
        for filename in os.listdir("gif_frames"):
            os.remove(f"gif_frames/{filename}")

    # load radar data
    print("\tLoading radar data...")
    radar_data, time_data = read_radar_data(path, start_date, end_date)
   
    # project radar data to WGS84
    grid = get_wgs84_grid() # (900, 900, 2)
    radar_data_wgs84 = np.zeros((radar_data.shape[0], 900, 900))
    for i in range(radar_data.shape[0]):
        radar_data_wgs84[i] = radar_data[i]

    # load german border
    border_x, border_y = get_german_border() # (B,), (B,)

    # change nan values to 0
    radar_data_wgs84 = np.nan_to_num(radar_data_wgs84)

    vmin = 0.0
    vmax = np.max(radar_data_wgs84[0])
    vmax_factor = 0.95

    # plot radar data
    plt.ion()
    for i in range(radar_data_wgs84.shape[0]):

        plt.clf()

        # update vmax
        if np.max(radar_data_wgs84[i]) > vmax:
            vmax = np.max(radar_data_wgs84[i])
        else:
            vmax = vmax * vmax_factor + np.max(radar_data_wgs84[i]) * (1 - vmax_factor)

        plt.imshow(radar_data_wgs84[i], extent=[grid[0, 0, 1], grid[0, -1, 1], grid[0, 0, 0], grid[-1, 0, 0]], cmap="Blues", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(cmap="Blues")
        cbar.set_label("mm")

        plot_state_borders_on_axis(ax=plt, state_annotations=True, color="black")
        plt.title(str(time_data[i])[0:16].replace("T", " "))
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f °N'))
        plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f °E'))

        # scatter city of Wuppertal
        plt.scatter(7.150829, 51.256176, color="red", s=50, label="Wuppertal")


        # set aspect ratio to be equal
        plt.gca().set_aspect(1.43)

        # Calculate 10% of the range in each direction
        x_range = max(border_x) - min(border_x)
        y_range = max(border_y) - min(border_y)
        x_padding = x_range * 0.1
        y_padding = y_range * 0.1

        # Set x-axis limits with 10% padding on both sides
        plt.xlim([min(border_x) - x_padding, max(border_x) + x_padding])

        # Set y-axis limits with 10% padding on both sides
        plt.ylim([min(border_y) - y_padding, max(border_y) + y_padding])


        # save image to file
        plt.savefig(f"gif_frames/frame_{i:04d}.png")

        plt.pause(0.001)
    plt.ioff()
    plt.close()

    print(f"\tCreated {radar_data_wgs84.shape[0]} frames")

    # create gif from images
    images = []
    for filename in os.listdir("gif_frames"):
        images.append(imageio.imread(f"gif_frames/{filename}"))
    imageio.mimsave('radar_data.gif', images)

    print("\tCreated gif. Cleaning up...")

    # remove images
    for filename in os.listdir("gif_frames"):
        os.remove(f"gif_frames/{filename}")

    # remove directory
    os.rmdir("gif_frames")
    print("\tDone!")


def plot_range():
    path = Path("F:\\radar_data")
    start_date = datetime(2023, 12, 9)
    end_date = datetime(2023, 12, 10)

    # load radar data
    radar_data, time_data = read_radar_data(path, start_date, end_date)

    # change NAN values to 0
    radar_data = np.nan_to_num(radar_data)
   
    # project radar data to WGS84
    grid = get_wgs84_grid() # (900, 900, 2)
    radar_data_wgs84 = np.zeros((radar_data.shape[0], 900, 900))
    for i in range(radar_data.shape[0]):
        radar_data_wgs84[i] = radar_data[i]

    # find the time where the maximum rainfall was recorded
    # if it is not i then continue
    max_mean = 0
    for j in range(radar_data_wgs84.shape[0]):
        if np.mean(radar_data_wgs84[j]) > max_mean:
            max_mean = np.mean(radar_data_wgs84[j])
            max_index = j
    print(max_index)

    # plot radar data
    vmin = 0.0
    vmax = np.max(radar_data_wgs84[max_index])
    fig = plt.figure(figsize=(8, 8))

    for i in range(radar_data_wgs84.shape[0]):

        if i != max_index:
            continue

        plt.imshow(radar_data_wgs84[i], extent=[grid[0, 0, 1], grid[0, -1, 1], grid[0, 0, 0], grid[-1, 0, 0]], cmap="Blues", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(cmap="Blues")
        cbar.set_label("mm")

        plot_german_border_on_axis(ax=plt, color="black")
        plt.title(str(time_data[i])[0:16].replace("T", " "))
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f °N'))
        plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f °E'))

        # scatter city of Wuppertal
        plt.scatter(7.150829, 51.256176, color="red", s=50, label="Wuppertal")


        # set aspect ratio to be equal
        plt.gca().set_aspect(1.43)

        plt.show()


def compute_precipitation(year_month_tuple):
    year, month = year_month_tuple
    """Worker function to compute average precipitation for a given year and month."""
    start_date = datetime(year, month, 1)
    end_date = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

    # Load radar data
    radar_data, _ = read_radar_data(start_date, end_date)
    #radar_data = np.nan_to_num(radar_data)

    # Compute total precipitation
    radar_data[:, get_germany_mask()] = np.nan # aplly mask
    inner = np.nanmean(radar_data, axis=(1, 2), dtype=np.float64) # mean over space
    avg_precip = np.nansum(inner) # sum over time

    # Comute total precipitation of NRW
    radar_data[:, get_NRW_mask()] = np.nan # aplly mask
    inner = np.nanmean(radar_data, axis=(1, 2), dtype=np.float64) # mean over space
    nrw_precip = np.nansum(inner) # sum over time

    print(f"Year: {year}, Month: {month}, Avg. Precipitation: {avg_precip:.2f} mm, NRW Precipitation: {nrw_precip:.2f} mm")
    return year, month, avg_precip, nrw_precip
def plot_avg_precipitation_per_month(max_parallel_processes=6):
    """Compute and plot average monthly precipitation per square meter."""
    start_time = time.perf_counter()

    # Prepare tasks
    years = range(2006, 2024)
    months = range(1, 13)
    tasks = [(year, month) for year in years for month in months]

    #result = compute_precipitation(tasks[0])
    #print(result)
    #exit()

    # Process tasks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_parallel_processes) as executor:
        results = list(executor.map(compute_precipitation, tasks))

    # Aggregate results
    data = [[] for _ in range(12)]
    data_nrw = [[] for _ in range(12)]
    for _, month, value, value_nrw in results:
        data[month - 1].append(value)
        data_nrw[month - 1].append(value_nrw)
    data = np.array(data)
    data_nrw = np.array(data_nrw)
    print(f'Elapsed time: {time.perf_counter() - start_time:.2f} s')
    
    # plot average precipitation per month
    months_str = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.boxplot(data.T, labels=months_str)
    plt.plot(months, np.mean(data_nrw, axis=1), label='NRW mean', color='red')
    plt.xlabel("Month")
    plt.ylabel("precipitation [mm]")
    plt.title("Average monthly precipitation per square meter in Germany")
    plt.legend()
    plt.show()

    # plot average precipitation per year as bar chart
    years = [str(year) for year in range(2006, 2024)]
    fig = plt.figure(figsize=(10, 5))
    plt.bar(years, np.sum(data, axis=0), color="blue", label="Germany")
    plt.plot(years, np.sum(data_nrw, axis=0), color="red", label="NRW")
    plt.xlabel("Year")
    plt.ylabel("precipitation [mm]")
    plt.title("Average yearly precipitation per square meter")
    plt.legend()
    plt.show()


def plot_average_percipitation_on_map():
    data = []
    for year in range(2006, 2022+1):
        for month in range(1, 12+1):
            try:
                path = Path(f"cum_precipitation_per_moth/{year}/{month}.npy")
                data.append(np.load(path))
            except Exception as e:
                print(e)
                continue
    data = np.array(data)
    print(data.shape)

    data = np.mean(data, axis=0)
    print(data.shape)

    # limit values to 200
    data[data > 200] = 200

    # project radar data to WGS84
    grid = get_wgs84_grid() # (900, 900, 2)
    
    # load german border
    border_x, border_y = get_german_border() # (B,), (B,)

    # plot radar data
    vmin = 0.0
    vmax = np.max(data)
    fig = plt.figure(figsize=(8, 8))

    plt.imshow(data, extent=[grid[0, 0, 1], grid[0, -1, 1], grid[0, 0, 0], grid[-1, 0, 0]], cmap="Blues", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cmap="Blues")
    cbar.set_label("mm")

    #plt.plot(border_x, border_y, color="black")
    plot_state_borders_on_axis(ax=plt, state_annotations=False, color="black")
    plt.title("Average precipitation per month")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f °N'))
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f °E'))

    # scatter city of Wuppertal
    plt.scatter(7.150829, 51.256176, color="red", s=50, label="Wuppertal")

    # set aspect ratio to be equal
    plt.gca().set_aspect(1.43)

    # Calculate 10% of the range in each direction
    x_range = max(border_x) - min(border_x)
    y_range = max(border_y) - min(border_y)
    x_padding = x_range * 0.1
    y_padding = y_range * 0.1

    # Set x-axis limits with 10% padding on both sides
    plt.xlim([min(border_x) - x_padding, max(border_x) + x_padding])
    plt.ylim([min(border_y) - y_padding, max(border_y) + y_padding])

    plt.show()


def compute_daily_precipitation(year_month_tuple):
    year, month = year_month_tuple
    """Worker function to compute average precipitation for a given year and month."""
    start_date = datetime(year, month, 1)
    end_date = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

    # Load radar data
    radar_data, _ = read_radar_data(start_date, end_date)
    #radar_data = np.nan_to_num(radar_data)

    # Compute total precipitation
    radar_data[:, get_germany_mask()] = np.nan # aplly mask
    inner = np.nanmean(radar_data, axis=(1, 2), dtype=np.float64) # mean over space

    # cut some data at the end off, so that we have a multiple of 24
    tmp = inner.shape[0] % 24
    if tmp != 0:
        inner = inner[:-tmp]

    # sum over 24 consecutive hours
    daily_precip = np.nansum(inner.reshape(-1, 24), axis=1) # sum over time
    print(f"Year: {year}, Month: {month}, Daily Precipitation: {daily_precip}")
    return year, month, daily_precip
def plot_daily_precipitation(max_parallel_processes=6):
    """Compute and plot average monthly precipitation per square meter."""
    start_time = time.perf_counter()

    # Prepare tasks
    years = range(2006, 2024)
    months = range(1, 13)
    tasks = [(year, month) for year in years for month in months]

    #result = compute_daily_precipitation(tasks[0])
    #print(result)
    #exit()

    # Process tasks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_parallel_processes) as executor:
        results = list(executor.map(compute_daily_precipitation, tasks))

    # Aggregate results
    data = []
    for y, m, values in results:
        data += list(values)
        print(f"Year: {y}, \tMonth: {m}, \tDaily Precipitation: {values}")
    data = np.array(data)

    print(f'Elapsed time: {time.perf_counter() - start_time:.2f} s')

    # save data to file
    np.save("daily_precipitation.npy", data)

    # plot daily precipitation
    plt.plot(data)
    plt.xlabel("Day")
    plt.ylabel("precipitation [mm]")
    plt.title("Daily precipitation in Germany")
    plt.show()


if __name__ == "__main__":

    plot_daily_precipitation()
