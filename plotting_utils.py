import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import json
from scipy.signal import convolve2d


def plot_state_borders_on_axis(ax=plt, state_annotations=True, color="black", only_NRW=False):
    states_abbreviation_lookup = {
        "Baden-Württemberg": "BW",
        "Bayern": "BY",
        "Berlin": "BE",
        "Brandenburg": "BB",
        "Bremen": "HB",
        "Hamburg": "HH",
        "Hessen": "HE",
        "Mecklenburg-Vorpommern": "MV",
        "Niedersachsen": "NI",
        "Nordrhein-Westfalen": "NRW",
        "Rheinland-Pfalz": "RP",
        "Saarland": "SL",
        "Sachsen": "SN",
        "Sachsen-Anhalt": "ST",
        "Schleswig-Holstein": "SH",
        "Thüringen": "TH"
    }
    path = Path("borders/german_states.json")
    with open(path, mode="r", encoding="utf-8") as file:
        data = json.load(file)
    data = data["features"]
    for state_data in data:
        state_name = state_data["properties"]["name"]
        if only_NRW and state_name != 'Nordrhein-Westfalen':
            continue
        poly_type = state_data["geometry"]["type"]
        state_polygons = state_data["geometry"]["coordinates"]
        if poly_type == 'Polygon':
            for polygon in state_polygons:
                x, y = np.array(polygon).T
                ax.plot(x, y, color=color)
                if state_annotations:
                    ax.annotate(states_abbreviation_lookup[state_name], (np.mean(x), np.mean(y)), color=color, ha='center', va='center')
        elif poly_type == 'MultiPolygon':
            tmp = True
            for polygon in state_polygons:
                for poly in polygon:
                    x, y = np.array(poly).T
                    ax.plot(x, y, color=color)
                    if tmp and state_annotations:
                        ax.annotate(states_abbreviation_lookup[state_name], (np.mean(x), np.mean(y)), color=color, ha='center', va='center')
                    tmp = False


def plot_german_border_on_axis(ax=plt, color="black"):
    path = Path("borders/germany.json")
    with open(path, "r") as file:
        data = json.load(file)
    coordinates = data["features"][0]["geometry"]["coordinates"][0]
    coordinates = np.array(coordinates)[0] # (2374, 2)
    x, y = coordinates[:, 0], coordinates[:, 1]
    ax.plot(x, y, color=color)


def get_german_border():
    path = Path("borders/germany.json")
    with open(path, "r") as file:
        data = json.load(file)
    coordinates = data["features"][0]["geometry"]["coordinates"][0]
    coordinates = np.array(coordinates)[0] # (2374, 2)
    x, y = coordinates[:, 0], coordinates[:, 1]
    return x, y


def plot_topological_map():
    asc_file = "topology/dgm1000/dgm1000_utm32s.asc"
    prj_file = "topology/dgm1000/dgm1000_utm32s.prj"

    with open(prj_file, "r") as file:
        prj = file.read()

    data = np.loadtxt(asc_file, skiprows=6)
    data[data < -400] = 0
    data[data > 1000] = 1000

    # smooth data by averaging over 3x3 kernel
    kernel_size = 50
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    data = convolve2d(data, kernel, mode="same")

    plt.hist(data.flatten(), bins=100)
    plt.show()

    # create 3d plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(0, data.shape[1])
    y = np.arange(0, data.shape[0])
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, data, cmap="terrain")
    ax.set_zlim(0, 1000)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

if __name__ == '__main__':
    plot_topological_map()