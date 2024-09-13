import copy

import astropy.units as u
import numpy as np
import sunpy
import sunpy.map
from astropy.units import UnitsError
from astropy.visualization import make_lupton_rgb
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sunpy.map import Map, all_coordinates_from_map
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.optimize import minimize

m171 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits")
m193 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits")
m211 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits")

# From old IDL code
threshold_171v193 = 0.6357
threshold_171v211 = 0.7
threshold_193v211 = 1.5102

'''I tried a new method of calculating the slopes in which I made a path out of the bin edges and a second path for the 
segmentation line to identify the bins that the line passed through. I then added up the total counts and made a funtion 
to vary and optimize the slope by minimizing the total counts. I don't think my optimization section works correctly so maybe
you could take a look?'''

def find_edges(wave1: int, wave2: int):
    map1 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0" + str(wave1) + ".fits")
    map2 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0" + str(wave2) + ".fits")
    # Since the data are taken at similar times neglect any coordinate changes so just use 171 maps coordinates
    coords = all_coordinates_from_map(map1)
    disk_mask = (coords.Tx**2 + coords.Ty**2) ** 0.5 < map1.rsun_obs

    map1 = map1 / map1.exposure_time
    map2 = map2 / map2.exposure_time

    xx = np.linspace(0, 300, 500)

    cool_counts, cool_xedge, cool_yedge = np.histogram2d(
        map1.data[disk_mask].flatten(),
        map2.data[disk_mask].flatten(),
        bins=250,
        range=[[0, 300], [0, 300]],
        density=True,
    )

    xedges = np.array(cool_xedge)
    yedges = np.array(cool_yedge)
    return cool_counts, xedges, yedges

cool_hist, cool_xedges, cool_yedges = find_edges(171, 193)
warm_hist, warm_xedges, warm_yedges = find_edges(171, 211)
cool_threshold = threshold_171v193
warm_threshold = threshold_171v211

def bins_intersected(xedges: np.array, yedges: np.array, slope:float, threshold: int, hist: np.array):
    x = np.linspace(0, 300, 500)
    y = slope * (x**threshold)
    line_path = Path(list(zip(x, y)))
    counts = 0
    for i in range(len(xedges)  - 1):
        for j in range(len(yedges) - 1):
            bin_corners = [
                (xedges[i], yedges[j]),
                (xedges[i + 1], yedges[j]),
                (xedges[i + 1], yedges[j + 1]),
                (xedges[i], yedges[j + 1])
            ] 
            bin_path = Path(bin_corners)
            if line_path.intersects_path(bin_path):
                counts += hist[j, i]
    return counts

def objective_function(slope: float, xedges: np.array, yedges: np.array, threshold: float, hist: np.array):
    return bins_intersected(xedges, yedges, slope, threshold, hist)

initial_guess_cool = 2.25
initial_guess_warm = .5
bounds_cool = [(2, 2.5)]
bounds_warm = [(.3, .75)]
cool_result = minimize(objective_function, x0 = initial_guess_cool, args=(cool_xedges, cool_yedges, cool_threshold, cool_hist), method = 'Powell', bounds = bounds_cool)
warm_result = minimize(objective_function, x0 = initial_guess_warm, args=(warm_xedges, warm_yedges, warm_threshold, warm_hist), method = 'Powell', bounds = bounds_warm)


cool_optimal_slope = warm_result.x[0]
cool_minimized_counts = warm_result.fun

def segmenting_plots(scale_cold: float, scale_warm: float, scale_hot: float):
    m171 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits")
    m193 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits")
    m211 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits")

    m171_orig = copy.deepcopy(m171)
    m193_orig = copy.deepcopy(m193)
    m211_orig = copy.deepcopy(m211)

    # Since the data are taken at similar times neglect any coordinate changes so just use 171 maps coordinates
    coords = all_coordinates_from_map(m171)
    disk_mask = (coords.Tx**2 + coords.Ty**2) ** 0.5 < m171.rsun_obs

    m171 = m171 / m171.exposure_time
    m193 = m193 / m193.exposure_time
    m211 = m211 / m211.exposure_time

    xx = np.linspace(0, 300, 500)

    fig, axes = plt.subplot_mosaic(
        [["cool_hist"], ["warm_hist"], ["hot_hist"]],
        layout="constrained",
        figsize=(3, 5),
    )

    # 171 v 193
    cool_counts, *cool_bins = axes["cool_hist"].hist2d(
        m171.data[disk_mask].flatten(),
        m193.data[disk_mask].flatten(),
        bins=60,
        range=[[0, 300], [0, 300]],
        norm=LogNorm(),
        density=True,
    )



    axes["cool_hist"].set_facecolor("k")
    axes["cool_hist"].plot(xx, scale_cold * (xx**threshold_171v193), "w")

    # 171 v 211
    warm_counts, *warm_bins = axes["warm_hist"].hist2d(
        m171.data[disk_mask].flatten(),
        m211.data[disk_mask].flatten(),
        bins=250,
        range=[[0, 300], [0, 300]],
        norm=LogNorm(),
        density=True,
    )
    # Finding the indices of nonzero counts
    non_zero_warm = np.where(warm_counts > 0)

    # Finding the corresponding minimum y-index
    min_y_index = np.min(non_zero_warm[1])

    # Map the index to the actual y-value
    min_y_warm = warm_bins[1][min_y_index]
    axes["warm_hist"].set_ylim(0, 100)
    axes["warm_hist"].set_facecolor("k")
    axes["warm_hist"].plot(xx, (scale_warm * (xx**threshold_171v211)) + min_y_warm, "w")

    # 193 v 311
    hot_counts, *hot_bins = axes["hot_hist"].hist2d(
        m193.data[disk_mask].flatten(),
        m211.data[disk_mask].flatten(),
        bins=250,
        range=[[0, 300], [0, 300]],
        norm=LogNorm(),
        density=True,
    )
    # Finding the indices of nonzero counts
    non_zero_hot = np.where(hot_counts > 0)

    # Finding the corresponding minimum y-index
    min_y_index = np.min(non_zero_hot[1])

    # Map the index to the actual y-value
    min_y_hot = hot_bins[1][min_y_index]

    axes["hot_hist"].set_ylim(0, 100)
    axes["hot_hist"].set_facecolor("k")
    axes["hot_hist"].plot(xx, scale_hot * (xx**-threshold_193v211) + min_y_hot, "w")

    plt.show()


segmenting_plots(2.025, .5, 2600)


def clip_scale(map1: sunpy.map, clip1: float, clip2: float, clip3: float, scale: float):
    map_clipped = np.clip(np.log10(map1.data), clip1, clip3)
    map_clipped_scaled = ((map_clipped - clip1) / clip2) * scale
    return map_clipped_scaled


cs_171 = clip_scale(m171, 1.2, 2.7, 3.9, 255)
cs_193 = clip_scale(m193, 1.4, 1.6, 3.0, 255)
cs_211 = clip_scale(m211, 0.8, 1.9, 2.7, 255)


def create_mask(scale1: float, scale2: float, scale3: float):
    mask1 = (cs_171 / cs_211) >= (np.mean(m171.data) * scale1) / np.mean(m211.data)
    mask2 = (cs_211 + cs_193) < (scale2 * (np.mean(m193.data) + np.mean(m211.data)))
    mask3 = (cs_171 / cs_193) >= ((np.mean(m171.data) * scale3) / np.mean(m193.data))
    return mask1, mask2, mask3


mask211_171, mask211_193, mask171_193 = create_mask(0.6357, 0.7, 1.5102)

tri_color_img = make_lupton_rgb(cs_171, cs_193, cs_211, Q=10, stretch=50)
comb_mask = mask211_171 * mask211_193 * mask171_193


fig, axes = plt.subplot_mosaic([["tri", "comb_mask"]], layout="constrained", figsize=(6, 3))

axes["tri"].imshow(tri_color_img, origin="lower")
axes["comb_mask"].imshow(comb_mask, origin="lower")

plt.show()


def create_contours(mask1: np.ndarray, mask2: np.ndarray, mask3: np.ndarray):
    mask_map = Map(((mask1 * mask2 * mask3).astype(int), m171.meta))
    try:
        contours = mask_map.contour(0.5 / u.s)
    except UnitsError:
        contours = mask_map.contour(50 * u.percent)

    contours = sorted(contours, key=lambda x: x.size, reverse=True)

    fig, axes = plt.subplot_mosaic(
        [["seg"]], layout="constrained", figsize=(6, 3), subplot_kw={"projection": m171}
    )
    m171.plot(axes=axes["seg"])
    axes["seg"].imshow(tri_color_img)

    # For the moment just plot to top 5 contours based on "size" for contour
    for contour in contours[:6]:
        axes["seg"].plot_coord(contour, color="w", linewidth=0.5)
    plt.show()


create_contours(mask211_171, mask211_193, mask171_193)
