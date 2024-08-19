"""
=============
Paper Figures
=============

Recreation of figures from the original paper [Garton2018]_.

.. [Garton2018] https://www.swsc-journal.org/articles/swsc/pdf/2018/01/swsc170041.pdf

Srart of with imports
"""

import copy

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import make_lupton_rgb
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sunpy.map import Map, all_coordinates_from_map, pixelate_coord_path, sample_at_coords


#####################################################
#
# Figure 2
# --------
#
# .. note::
#     Figures 2 and 3 use data from a different date compared to later figures.
#

m94 =  Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/09/22/H1200/AIA20160922_1200_0094.fits")
m131 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/09/22/H1200/AIA20160922_1200_0131.fits")
m171 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/09/22/H1200/AIA20160922_1200_0171.fits")
m193 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/09/22/H1200/AIA20160922_1200_0193.fits")
m211 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/09/22/H1200/AIA20160922_1200_0211.fits")
m335 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/09/22/H1200/AIA20160922_1200_0335.fits")

fig, axes = plt.subplot_mosaic(
    [["131", "171"], ["193", "211"], ["335", "94"]],
    layout="constrained",
    figsize=(6, 9),
    per_subplot_kw={
        "131": {"projection": m131},
        "171": {"projection": m171},
        "193": {"projection": m193},
        "211": {"projection": m211},
        "335": {"projection": m335},
        "94": {"projection": m94},
    },
)

for m in [m131, m171, m193, m211, m335, m94]:
    wave_str = f"{m.wavelength.value:0.0f}"
    line_coords = SkyCoord([-200, 0], [-100, -100], unit=(u.arcsec, u.arcsec),
                       frame=m.coordinate_frame)
    m.plot(axes=axes[wave_str], clip_interval=(1, 99)*u.percent)
    axes[wave_str].plot_coord(line_coords, 'w', linewidth=1.5)

#####################################################
#
# Figure 3
# --------
#

fig, axes = plt.subplot_mosaic(
    [["131"], ["171"], ["193"], ["211"], ["335"], ['94']],
    layout="constrained",
    figsize=(6, 9),
    sharex=True
)
for m in [m131, m171, m193, m211, m335, m94]:
    line_coords = SkyCoord([-200, 0], [-100, -100], unit=(u.arcsec, u.arcsec),
                           frame=m.coordinate_frame)
    x, y = m.world_to_pixel(line_coords)
    xmin, xmax = np.round(x).value.astype(int)
    ymin, ymax = np.round(y).value.astype(int)
    intensity_coords = pixelate_coord_path(m, line_coords, bresenham=True)
    intensity1 = sample_at_coords(m, intensity_coords)
    intensity2 = m.data[ymin, xmin:xmax+1]
    contrast = (intensity2.max() - intensity2.min()) / (intensity2.max() + intensity2.min())
    axes[format(m.wavelength.value, '0.0f')].plot(intensity_coords.Tx, intensity1)
    axes[format(m.wavelength.value, '0.0f')].plot(intensity_coords.Tx, intensity2, '--')
    axes[format(m.wavelength.value, '0.0f')].set_title(f'{m.wavelength: 0.0f}, Contrast: {contrast:0.2f}')


#####################################################
#
# Figure 4
# --------
#

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
# m171.data[~disk_mask] = np.nan
# m193.data[~disk_mask] = np.nan
# m211.data[~disk_mask] = np.nan


fig, axes = plt.subplot_mosaic(
    [["cool_scat", "cool_hist"], ["warm_scat", "warm_hist"], ["hot_scat", "hot_hist"]],
    layout="constrained",
    figsize=(6, 9),
)

axes["cool_scat"].scatter(m171.data, m193.data, alpha=0.2, edgecolor="none", s=2.5)
axes["cool_scat"].set_xlim(0, 2000)
axes["cool_scat"].set_ylim(0, 2000)
cool_counts, *cool_bins = axes["cool_hist"].hist2d(
    m171.data[disk_mask].flatten(),
    m193.data[disk_mask].flatten(),
    bins=150,
    range=[[0, 300], [0, 300]],
    norm=LogNorm(),
    density=True,
)
axes["cool_hist"].set_facecolor("k")
axes["cool_hist"].contour(cool_counts.T, np.linspace(1e-6, 1e-4, 5), cmap="Greys", extent=(0, 300, 0, 300))

axes["warm_scat"].scatter(m171.data, m211.data, alpha=0.2, edgecolor="none", s=2.5)
axes["warm_scat"].set_xlim(0, 2000)
axes["warm_scat"].set_ylim(0, 2000)
warm_counts, *warm_bins = axes["warm_hist"].hist2d(
    m171.data[disk_mask].flatten(),
    m211.data[disk_mask].flatten(),
    bins=250,
    range=[[0, 300], [0, 300]],
    norm=LogNorm(),
    density=True,
)
axes["warm_hist"].set_ylim(0, 100)
axes["warm_hist"].set_facecolor("k")
axes["warm_hist"].contour(warm_counts.T, np.linspace(1e-5, 5e-4, 5), cmap="Greys", extent=(0, 300, 0, 300))

axes["hot_scat"].scatter(m211.data, m193.data, alpha=0.2, edgecolor="none", s=2.5)
axes["hot_scat"].set_xlim(0, 2000)
axes["hot_scat"].set_ylim(0, 2000)
hot_counts, *hot_bins = axes["hot_hist"].hist2d(
    m193.data[disk_mask].flatten(),
    m211.data[disk_mask].flatten(),
    bins=250,
    range=[[0, 300], [0, 300]],
    norm=LogNorm(),
    density=True,
)
axes["hot_hist"].set_ylim(0, 100)
axes["hot_hist"].set_facecolor("k")
axes["hot_hist"].contour(hot_counts.T, np.linspace(1e-5, 1e-3, 5), cmap="Greys", extent=(0, 300, 0, 300))


#####################################################
#
# Figure 5
# --------
#
# Essentially the same as Fig 4 left colum in log scale version of right column.
#
# .. note::
#     The fit lines are currently arbitrary as numbers are not stated in the paper need to add fitting
#

xx = np.linspace(0, 300, 500)

fig, axes = plt.subplot_mosaic(
    [["cool_log", "cool_hist"], ["warm_log", "warm_hist"], ["hot_log", "hot_hist"]],
    layout="constrained",
    figsize=(6, 9),
)

# 171 v 193
axes["cool_log"].hist2d(
    m171.data[disk_mask].flatten(),
    m193.data[disk_mask].flatten(),
    bins=150,
    range=[[0, 300], [0, 300]],
    norm=LogNorm(),
    density=True,
)
axes["cool_log"].set_xlim(10, 150)
axes["cool_log"].set_xscale("log")
axes["cool_log"].set_ylim(10, 150)
axes["cool_log"].set_yscale("log")
axes["cool_log"].set_facecolor("k")
axes["cool_log"].plot(xx, xx**0.7, "w")


cool_counts, *cool_bins = axes["cool_hist"].hist2d(
    m171.data[disk_mask].flatten(),
    m193.data[disk_mask].flatten(),
    bins=150,
    range=[[0, 300], [0, 300]],
    norm=LogNorm(),
    density=True,
)
axes["cool_hist"].set_facecolor("k")
axes["cool_hist"].plot(xx, xx**0.7, "w")


# 171 v 211
axes["warm_log"].hist2d(
    m171.data[disk_mask].flatten(),
    m211.data[disk_mask].flatten(),
    bins=250,
    range=[[0, 300], [0, 300]],
    norm=LogNorm(),
    density=True,
)
axes["warm_log"].set_xlim(10, 150)
axes["warm_log"].set_xscale("log")
axes["warm_log"].set_ylim(3, 100)
axes["warm_log"].set_yscale("log")
axes["warm_log"].plot(xx, xx**0.6, "w")

warm_counts, *warm_bins = axes["warm_hist"].hist2d(
    m171.data[disk_mask].flatten(),
    m211.data[disk_mask].flatten(),
    bins=250,
    range=[[0, 300], [0, 300]],
    norm=LogNorm(),
    density=True,
)
axes["warm_hist"].set_ylim(0, 100)
axes["warm_hist"].set_facecolor("k")
axes["warm_hist"].plot(xx, xx**0.6, "w")

# 193 v 311
axes["hot_log"].hist2d(
    m193.data[disk_mask].flatten(),
    m211.data[disk_mask].flatten(),
    bins=250,
    range=[[0, 300], [0, 300]],
    norm=LogNorm(),
    density=True,
)
axes["hot_log"].set_xlim(10, 100)
axes["hot_log"].set_xscale("log")
axes["hot_log"].set_ylim(2, 100)
axes["hot_log"].set_yscale("log")
axes["hot_log"].plot(xx, xx**0.7, "w")

hot_counts, *hot_bins = axes["hot_hist"].hist2d(
    m193.data[disk_mask].flatten(),
    m211.data[disk_mask].flatten(),
    bins=250,
    range=[[0, 300], [0, 300]],
    norm=LogNorm(),
    density=True,
)
axes["hot_hist"].set_ylim(0, 100)
axes["hot_hist"].set_facecolor("k")
axes["hot_hist"].plot(xx, xx**0.7, "w")


#####################################################
#
# Figure 6
# --------
#
fig, axes = plt.subplot_mosaic(
    [["tri", "193_171"], ["211_171", "211_193"]], layout="constrained", figsize=(6, 6)
)

d171_clipped = np.clip(np.log10(m171.data), 1.2, 3.9)
d171_clipped_scaled = ((d171_clipped - 1.2) / 2.7) * 255

d193_clipped = np.clip(np.log10(m193.data), 1.4, 3.0)
d193_clipped_scaled = ((d193_clipped - 1.4) / 1.6) * 255

d211_clipped = np.clip(np.log10(m211.data), 0.8, 2.7)
d211_clipped_scaled = ((d211_clipped - 0.8) / 1.9) * 255

tri_color_img = make_lupton_rgb(
    d171_clipped_scaled, d193_clipped_scaled, d211_clipped_scaled, Q=10, stretch=50
)

mask_171_211 = (d171_clipped_scaled / d211_clipped_scaled) >= (
    (np.mean(m171.data) * 0.6357) / np.mean(m211.data)
)
mask_211_193 = (d211_clipped_scaled + d193_clipped_scaled) < (0.7 * (np.mean(m193.data) + np.mean(m211.data)))
mask_171_193 = (d171_clipped_scaled / d193_clipped_scaled) >= (
    (np.mean(m171.data) * 1.5102) / np.mean(m193.data)
)

axes["tri"].imshow(tri_color_img, origin="lower")
axes["193_171"].imshow(mask_171_193, origin="lower")
axes["211_171"].imshow(mask_171_211, origin="lower")
axes["211_193"].imshow(mask_211_193, origin="lower")


#####################################################
#
# Figure 7
# --------
#
fig, axes = plt.subplot_mosaic([["tri", "comb_mask"]], layout="constrained", figsize=(6, 3))

axes["tri"].imshow(tri_color_img, origin="lower")
axes["comb_mask"].imshow(mask_171_193 * mask_171_211 * mask_211_193, origin="lower")


#####################################################
#
# Figure 8
# --------
#

mask_map = Map(((mask_171_193 * mask_171_211 * mask_211_193).astype(int), m171.meta))
contours = mask_map.contour(0.5/u.s)
contours = sorted(contours, key=lambda x: x.size, reverse=True)

fig, axes = plt.subplot_mosaic([['seg']], layout="constrained", figsize=(6, 3), subplot_kw={'projection': m171})
m171.plot(axes=axes['seg'])
axes['seg'].imshow(tri_color_img)

# For the moment just plot to top 5 contours based on "size" for contour
for contour in contours[:6]:
    axes["seg"].plot_coord(contour, color='w')