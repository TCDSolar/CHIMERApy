"""
=============
Paper Figures
=============

Recreation of figures from the original paper [Garton2018]_.

.. [Garton2018] https://www.swsc-journal.org/articles/swsc/pdf/2018/01/swsc170041.pdf

Start of with imports
"""

import copy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from sunpy.map import (
    Map,
    all_coordinates_from_map,
    coordinate_is_on_solar_disk,
    pixelate_coord_path,
    sample_at_coords,
)

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.units import UnitsError
from astropy.visualization import make_lupton_rgb

# %%
#
# Figure 2
# --------
#
# A summary plot showing SDO/AIA 131Å, 171Å, 193Å, 211Å, 335Å and 94Å observations from 2016-09-22 at 12:00 UT.
# A small equatorial coronal hole (CH) is visible close to disk center, it is more visible in some wavelengths than
# others.
#
# .. note::
#     Figures 2 and 3 use data from a different date compared to later figures.
#

m94 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/09/22/H1200/AIA20160922_1200_0094.fits")
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
plt.close('all')
temps = {
    "131": 0.4,
    "171": 0.7,
    "193": 1.2,
    "211": 2.0,
    "335": 2.5,
    "94": 6.3,
}
for m in [m131, m171, m193, m211, m335, m94]:
    wave_str = f"{m.wavelength.value:0.0f}"
    line_coords = SkyCoord([-200, 0], [-100, -100], unit=(u.arcsec, u.arcsec), frame=m.coordinate_frame)
    m.plot(axes=axes[wave_str], clip_interval=(0.1, 99.9) * u.percent)
    axes[wave_str].set_title(f"{wave_str} Å {temps[wave_str]} MK")
    axes[wave_str].plot_coord(line_coords, "w", linewidth=1.5)
    axes[wave_str].set_facecolor("k")

# %%
#
# Figure 3
# --------
#
# Intensity cuts across the CH along the path indicated in Figure 1 for each wavelength.
# The Michelson contrast for each intensity cut is calculated via :math:`C_{M} \frac{I_{Max}-I_{Min}}{I_{Max}+I_{Min}}`
# and display in the plot title.
#

for m in [m131, m171, m193, m211, m335, m94]:
    line_coords = SkyCoord([-200, 0], [-100, -100], unit=(u.arcsec, u.arcsec), frame=m.coordinate_frame)
    x, y = m.world_to_pixel(line_coords)
    xmin, xmax = np.round(x).value.astype(int)
    ymin, ymax = np.round(y).value.astype(int)
    intensity_coords = pixelate_coord_path(m, line_coords, bresenham=True)
    intensity1 = sample_at_coords(m, intensity_coords)
    contrast = (intensity1.max() - intensity1.min()) / (intensity1.max() + intensity1.min())
    axes[format(m.wavelength.value, "0.0f")].plot(intensity_coords.Tx, intensity1)
    axes[format(m.wavelength.value, "0.0f")].set_title(
        f"AIA {m.wavelength.value: 0.0f} Å, Contrast: {contrast:0.2f}"
    )


# %%
#
# Figure 4
# --------
#
# Figure 4 shows intensity ratio plots for three wavelength combinations 171Å vs 193Å, 171Å vs 211Å,
# and finally 211Å vs 193Å. The plots in the left column are simple scatter plots of the intensity of one wavelength
# against the other. The plots on the right are 2D histograms constructed using the same data focusing on the lower
# intensity regions marked by the red rectangles.

m171 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits")
m193 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits")
m211 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits")

m171_orig = copy.deepcopy(m171)
m193_orig = copy.deepcopy(m193)
m211_orig = copy.deepcopy(m211)

# Since the data are taken at similar times neglect any coordinate changes so just use 171 maps coordinates
coords = all_coordinates_from_map(m171)
disk_mask = coordinate_is_on_solar_disk(coords)

m171 = m171 / m171.exposure_time
m193 = m193 / m193.exposure_time
m211 = m211 / m211.exposure_time

# # Smooth the data with 4x4 median filter (not need on NRT data)
# m171.data[:, :] = medfilt2d(m171.data, 5)
# m193.data[:, :] = medfilt2d(m193.data, 5)
# m211.data[:, :] = medfilt2d(m211.data, 5)




# %%
#
# Figure 5
# --------
#
# Show a smaller region around the low intensities in both log (left) and linear (right) scale. Also show are the
# delineation between candidate CH and non-CH intensities.
#
# .. note::
#     The delineations are currently :math:`\chi` by eye as the constants/intercepts are not stated in the paper.
#

# From old IDL code
threshold_171v193 = 0.6357
threshold_171v211 = 0.7
threshold_193v211 = 1.5102

# Equations of the form I_y = c * I_x**thres c by eye
xx = np.linspace(0, 300, 500)
fit171v193 = 2.25 * xx**threshold_171v193
fit171v211 = 0.5 * xx**threshold_171v211
fit193v211 = 2600 * xx**-threshold_193v211


# %%
#
# Figure 6
# --------
#
# Figure 6 shows the log transformed and clipped 171, 193, and 211 as a tri-color image (top left). Candidate CH masks
# for each intensity ratio pair are also shown 193Å - 171Å (top right), 211Å - 171Å (bottom left), and
# 211Å - 193Å (bottom right).
# These were obtained by thresholding ratios of the log transformed and clipped observations.

# From IDL code how they were obtained unclear maybe fitting of intensity valley?
d171_min, d171_max = 1.2, 3.9
d193_min, d193_max = 1.4, 3.0
d211_min, d211_max = 0.8, 2.7

d171_clipped = np.clip(np.log10(m171.data), d171_min, d171_max)
d171_clipped_scaled = ((d171_clipped - d171_min) / (d171_max - d171_min)) * 255

d193_clipped = np.clip(np.log10(m193.data), d193_min, d193_max)
d193_clipped_scaled = ((d193_clipped - d193_min) / (d193_max - d193_min)) * 255

d211_clipped = np.clip(np.log10(m211.data), d211_min, d211_max)
d211_clipped_scaled = ((d211_clipped - d211_min) / (d211_max - d211_min)) * 255

tri_color_img = make_lupton_rgb(
    d171_clipped_scaled, d193_clipped_scaled, d211_clipped_scaled, Q=10, stretch=50
)

mask_171_211 = (d171_clipped_scaled / d211_clipped_scaled) >= (
    (np.mean(m171.data[disk_mask]) * 0.6357) / np.mean(m211.data[disk_mask])
)

mask_211_193 = (d211_clipped_scaled + d193_clipped_scaled) < (
    0.7 * (np.mean(m193.data[disk_mask]) + np.mean(m211.data[disk_mask]))
)

mask_171_193 = (d171_clipped_scaled / d193_clipped_scaled) >= (
    (np.mean(m171.data[disk_mask]) * 1.5102) / np.mean(m193.data[disk_mask])
)

# %%
#
# Figure 7
# --------
#
# Figure 7 shows the log transformed and clipped 171, 193, and 211 as a tri-color image (left) and the final CH mask
# obtained by taking the product of the individual candidate masks.

# %%
#
# Figure 8
# --------
#
# Figure 8 shows the top 5 (by area) contours created from the CH mask over plotted on the same tri-color image as in
# Figures 6 and 7.

def final_mask(m171, tri_color_img, mask_171_211, mask_211_193, mask_171_193):
    mask_map = Map(((mask_171_193 * mask_171_211 * mask_211_193).astype(int), m171.meta))
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

final_mask(m171, tri_color_img, mask_171_211, mask_211_193, mask_171_193)