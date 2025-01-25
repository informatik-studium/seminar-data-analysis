from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from numba import cuda
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter
import os
import imageio.v2 as imageio

from utils import *
from plotting_utils import *


@cuda.jit
def CPWD_step(precipitation, prev_CPWD, result, decay, current_step):
    # this is a CUDA kernel. DO NOT CALL THIS FUNCTION DIRECTLY!

    # get the current position
    i, j = cuda.grid(2)

    # check if the current position is within the bounds of the array
    if i >= precipitation.shape[1] or j >= precipitation.shape[2]:
        return
    
    # check temporal bounds
    if current_step == 0 or current_step == precipitation.shape[0]:
        result[i, j] = precipitation[current_step, i, j]
        return
    
    # calculate the cumulative precipitation with decay
    prec = precipitation[current_step, i, j]
    if prec == np.nan:
        prec = np.float32(0)
    result[i, j] = prev_CPWD[i, j] * decay + precipitation[current_step, i, j]

def CPWD(decay=0.95, gif=False, save=False):

    # load precipitation data to the GPU
    precipitation = np.load('daily_precipitation.npy').astype(np.float32)
    precipitation_gpu = cuda.to_device(precipitation)

    # prepare the arrays for the cumulative precipitation with decay
    prev_CPWD = np.zeros((precipitation.shape[1], precipitation.shape[2]), dtype=np.float32)
    prev_CPWD_gpu = cuda.to_device(prev_CPWD)
    result = np.zeros_like(prev_CPWD, dtype=np.float32)
    result_gpu = cuda.to_device(result)
    decay = np.float32(decay)

    # create directory for gif frames
    if gif:
        if not os.path.exists("tmp/gif_frames"):
            os.makedirs("tmp/gif_frames")
        else:
            for filename in os.listdir("tmp/gif_frames"):
                os.remove(f"tmp/gif_frames/{filename}")
    
    # create directory for mean frames
    if save:
        if not os.path.exists("tmp/mean_frames"):
            os.makedirs("tmp/mean_frames")
        else:
            for filename in os.listdir("tmp/mean_frames"):
                os.remove(f"tmp/mean_frames/{filename}")

    # loop through the precipitation data and calculate the cumulative precipitation with decay
    start_datetime = datetime(2006, 1, 1)
    plt.rcParams["figure.figsize"] = (12, 10)
    plt.ion()
    grid = get_wgs84_grid()
    x = grid[:, :, 1]
    y = grid[:, :, 0]
    for i in range(precipitation.shape[0]):

        # compute next step on device
        CPWD_step[(32, 32), (32, 32)](precipitation_gpu, prev_CPWD_gpu, result_gpu, decay, i)
        result_gpu.copy_to_host(result)

        # blur with kernel
        result = gaussian_filter(result, sigma=3)

        # plot the cumulative precipitation with decay
        delta = timedelta(days=i)
        current_datetime = start_datetime + delta
        plt.clf()

        # save the current plot as a frame for the mean
        if save:
            np.save(f"tmp/mean_frames/mean_{i:04d}.npy", result)

        # use pcolormesh to plot the data
        plt.pcolormesh(
            x, y, result**2,
            cmap='turbo', # gnuplot
            shading='auto',
            vmin=0,
            vmax=25_000
        )
        plt.colorbar(label="mm²")

        plot_german_border_on_axis(plt, color='white')
        plot_state_borders_on_axis(plt, color='white', only_NRW=True, state_annotations=False)

        plt.scatter(7.150829, 51.256176, color="black", s=50, label="Wuppertal")
        plt.scatter(7.150829, 51.256176, color="white", s=30, label="Wuppertal")

        plt.gca().set_aspect(1.43)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f °N'))
        plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f °E'))

        plt.title(f'Cumulative Precipitation with Decay\n{current_datetime.strftime("%Y-%m-%d")}')

        if gif:
            plt.savefig(f"tmp/gif_frames/frame_{i:04d}.png")

        plt.pause(0.001)

        #if i == 365 + 6*30 + 16:
        #    plt.pause(100000)

        # update the previous cumulative precipitation with decay
        prev_CPWD_gpu.copy_to_host(prev_CPWD)
        prev_CPWD = result.copy()
        prev_CPWD_gpu = cuda.to_device(prev_CPWD)

    plt.ioff()
    #plt.show()
    plt.close()

    # create gif from images
    if gif:
        images = []
        for filename in os.listdir("tmp/gif_frames"):
            images.append(imageio.imread(f"tmp/gif_frames/{filename}"))
        imageio.mimsave('CPD.gif', images, fps=8)

        print("\tCreated gif. Cleaning up...")

        # remove images
        for filename in os.listdir("tmp/gif_frames"):
            os.remove(f"tmp/gif_frames/{filename}")

        # remove directory
        os.rmdir("tmp/gif_frames")

        # if tmp is empty, remove it
        if not os.listdir("tmp"):
            os.rmdir("tmp")

    if save:
        del precipitation, prev_CPWD, result # first we delete all data from RAM
        all = []
        for filename in os.listdir("tmp/mean_frames"):
            all.append(np.load(f"tmp/mean_frames/{filename}"))

        # delete mean frame files
        for filename in os.listdir("tmp/mean_frames"):
            os.remove(f"tmp/mean_frames/{filename}")

        # remove directory
        os.rmdir("tmp/mean_frames")

        # if tmp is empty, remove it
        if not os.listdir("tmp"):
            os.rmdir("tmp")

        all = np.array(all)
        mean = np.mean(all, axis=0)
        plt.pcolormesh(
            x, y, mean**2,
            cmap='turbo', # gnuplot
            shading='auto',
            vmin=0
        )
        plt.colorbar(label="mm²")
        plot_german_border_on_axis(plt, color='white')
        plot_state_borders_on_axis(plt, color='white', only_NRW=True, state_annotations=False)

        plt.scatter(7.150829, 51.256176, color="black", s=50, label="Wuppertal")
        plt.scatter(7.150829, 51.256176, color="white", s=30, label="Wuppertal")

        plt.gca().set_aspect(1.43)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f °N'))
        plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f °E'))

        plt.title(f'Cumulative Precipitation with Decay\nMEAN')

        plt.savefig("mean.png")




if __name__ == '__main__':
    CPWD(gif=True, save=True)