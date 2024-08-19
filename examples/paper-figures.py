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
from astropy.visualization import make_lupton_rgb
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sunpy.map import Map, all_coordinates_from_map


#####################################################
#
# Load the Data
# -------------------------
# * Normalise by exposure time
# * Set off disk pixels to nans
#

m94 = Map('https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0094.fits')
m131 = Map('https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0131.fits')
m171 = Map('https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits')
m193 = Map('https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits')
m211 = Map('https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits')
m335 = Map('https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0335.fits')

m171_orig = copy.deepcopy(m171)
m193_orig = copy.deepcopy(m193)
m211_orig = copy.deepcopy(m211)

#####################################################
#
# Figure 1
# --------
#
fig, axes = plt.subplot_mosaic([['131', '171'], ['193', '211'], ['335', '94']],
                               layout="constrained", figsize=(6, 9),
                               per_subplot_kw={'131': {'projection': m131},
                                               '171': {'projection': m171},
                                               '193': {'projection': m193},
                                               '211': {'projection': m211},
                                               '335': {'projection': m335},
                                               '94': {'projection': m94}})
m131.plot(axes=axes['131'])
m171.plot(axes=axes['171'])

m193.plot(axes=axes['193'])
m211.plot(axes=axes['211'])

m335.plot(axes=axes['335'])
m94.plot(axes=axes['94'])

del m94, m131, m335

#####################################################
#
# Figure 4
# --------
#

# Since the data are taken at similar times neglect any coordinate changes so just use 171 maps coordinates
coords = all_coordinates_from_map(m171)
disk_mask = (coords.Tx**2 + coords.Ty**2)**0.5 < m171.rsun_obs

m171 = m171 / m171.exposure_time
m193 = m193 / m193.exposure_time
m211 = m211 / m211.exposure_time
# m171.data[~disk_mask] = np.nan
# m193.data[~disk_mask] = np.nan
# m211.data[~disk_mask] = np.nan


fig, axes = plt.subplot_mosaic([['cool_scat', 'cool_hist'], ['warm_scat', 'warm_hist'],
                                ['hot_scat', 'hot_hist']],
                               layout="constrained", figsize=(6, 9))

axes['cool_scat'].scatter(m171.data, m193.data, alpha=0.2, edgecolor='none', s=2.5)
axes['cool_scat'].set_xlim(0, 2000)
axes['cool_scat'].set_ylim(0, 2000)
cool_counts, *cool_bins = axes['cool_hist'].hist2d(m171.data[disk_mask].flatten(), m193.data[disk_mask].flatten(),
                                                   bins=150, range=[[0, 300], [0, 300]], norm=LogNorm(), density=True)
axes['cool_hist'].set_facecolor('k')
axes['cool_hist'].contour(cool_counts.T, np.linspace(1e-6, 1e-4, 5), cmap='Greys',
                          extent=(0,300, 0, 300))

axes['warm_scat'].scatter(m171.data, m211.data, alpha=0.2, edgecolor='none', s=2.5)
axes['warm_scat'].set_xlim(0, 2000)
axes['warm_scat'].set_ylim(0, 2000)
warm_counts, *warm_bins = axes['warm_hist'].hist2d(m171.data[disk_mask].flatten(), m211.data[disk_mask].flatten(),
                                                   bins=250, range=[[0, 300], [0, 300]], norm=LogNorm(), density=True)
axes['warm_hist'].set_ylim(0, 100)
axes['warm_hist'].set_facecolor('k')
axes['warm_hist'].contour(warm_counts.T, np.linspace(1e-5, 5e-4, 5), cmap='Greys',
                          extent=(0, 300, 0, 300))

axes['hot_scat'].scatter(m211.data, m193.data, alpha=0.2, edgecolor='none', s=2.5)
axes['hot_scat'].set_xlim(0, 2000)
axes['hot_scat'].set_ylim(0, 2000)
hot_counts, *hot_bins = axes['hot_hist'].hist2d(m193.data[disk_mask].flatten(), m211.data[disk_mask].flatten(),
                                                bins=250, range=[[0, 300], [0, 300]], norm=LogNorm(), density=True)
axes['hot_hist'].set_ylim(0, 100)
axes['hot_hist'].set_facecolor('k')
axes['hot_hist'].contour(hot_counts.T, np.linspace(1e-5, 1e-3, 5), cmap='Greys',
                         extent=(0,300, 0, 300))

#####################################################
#
# Figure 5
# --------
#
# Essentially the same as Fig 4 left colum in log scale version of right column.
#
# .. note
#    Haven't actually done the fit as the number later on don't seem to match?

xx = np.linspace(0, 300, 500)

fig, axes = plt.subplot_mosaic([['cool_log', 'cool_hist'], ['warm_log', 'warm_hist'],
                                ['hot_log', 'hot_hist']],
                               layout="constrained", figsize=(6, 9))

# 171 v 193
axes['cool_log'].hist2d(m171.data[disk_mask].flatten(), m193.data[disk_mask].flatten(),
                        bins=150, range=[[0, 300], [0, 300]], norm=LogNorm(), density=True)
axes['cool_log'].set_xlim(10, 150)
axes['cool_log'].set_xscale('log')
axes['cool_log'].set_ylim(10, 150)
axes['cool_log'].set_yscale('log')
axes['cool_log'].set_facecolor('k')
axes['cool_log'].plot(xx, xx**0.7, 'w')


cool_counts, *cool_bins = axes['cool_hist'].hist2d(m171.data[disk_mask].flatten(), m193.data[disk_mask].flatten(),
                                                   bins=150, range=[[0, 300], [0, 300]], norm=LogNorm(), density=True)
axes['cool_hist'].set_facecolor('k')
axes['cool_hist'].plot(xx, xx**0.7, 'w')


# 171 v 211
axes['warm_log'].hist2d(m171.data[disk_mask].flatten(), m211.data[disk_mask].flatten(),
                        bins=250, range=[[0, 300], [0, 300]], norm=LogNorm(), density=True)
axes['warm_log'].set_xlim(10, 150)
axes['warm_log'].set_xscale('log')
axes['warm_log'].set_ylim(3, 100)
axes['warm_log'].set_yscale('log')
axes['warm_log'].plot(xx, xx**0.6, 'w')

warm_counts, *warm_bins = axes['warm_hist'].hist2d(m171.data[disk_mask].flatten(), m211.data[disk_mask].flatten(),
                                                   bins=250, range=[[0, 300], [0, 300]], norm=LogNorm(), density=True)
axes['warm_hist'].set_ylim(0, 100)
axes['warm_hist'].set_facecolor('k')
axes['warm_hist'].plot(xx, xx**0.6, 'w')

# 193 v 311
axes['hot_log'].hist2d(m193.data[disk_mask].flatten(), m211.data[disk_mask].flatten(),
                        bins=250, range=[[0, 300], [0, 300]], norm=LogNorm(), density=True)
axes['hot_log'].set_xlim(10, 100)
axes['hot_log'].set_xscale('log')
axes['hot_log'].set_ylim(2, 100)
axes['hot_log'].set_yscale('log')
axes['hot_log'].plot(xx, xx**0.7, 'w')

hot_counts, *hot_bins = axes['hot_hist'].hist2d(m193.data[disk_mask].flatten(), m211.data[disk_mask].flatten(),
                                                bins=250, range=[[0, 300], [0, 300]], norm=LogNorm(), density=True)
axes['hot_hist'].set_ylim(0, 100)
axes['hot_hist'].set_facecolor('k')
axes['hot_hist'].plot(xx, xx**0.7, 'w')


#####################################################
#
# Figure 6
# --------
#
fig, axes = plt.subplot_mosaic([['tri', '193_171'], ['211_171', '211_193']],
                               layout="constrained", figsize=(6, 3))

d171_clipped = np.clip(np.log10(m171.data), 1.2, 3.9)
d171_clipped_scaled = ((d171_clipped - 1.2) / 2.7)*255

d193_clipped = np.clip(np.log10(m193.data), 1.4, 3.0)
d193_clipped_scaled = ((d193_clipped - 1.4) / 1.6)*255

d211_clipped = np.clip(np.log10(m211.data), 0.8, 2.7)
d211_clipped_scaled = ((d211_clipped - 0.8) / 1.9)*255

tri_color_img = make_lupton_rgb(d171_clipped_scaled, d193_clipped_scaled, d211_clipped_scaled, Q=10, stretch=50)

mask_171_211 = (d171_clipped_scaled / d211_clipped_scaled) >= ((np.mean(m171.data)*0.6357) / np.mean(m211.data))
mask_211_193 = (d211_clipped_scaled + d193_clipped_scaled) < (0.7*(np.mean(m193.data) + np.mean(m211.data)))
mask_171_193 = (d171_clipped_scaled / d193_clipped_scaled) >= ((np.mean(m171.data)*1.5102) / np.mean(m193.data))

axes['tri'].imshow(tri_color_img, origin='lower')
axes['193_171'].imshow(mask_171_193, origin='lower')
axes['211_171'].imshow(mask_171_211, origin='lower')
axes['211_193'].imshow(mask_211_193, origin='lower')


#####################################################
#
# Figure 7
# --------
#
fig, axes = plt.subplot_mosaic([['tri', 'comb_mask']],
                               layout="constrained", figsize=(6, 3))

axes['tri'].imshow(tri_color_img, origin='lower')
axes['comb_mask'].imshow(mask_171_193*mask_171_211*mask_211_193, origin='lower')