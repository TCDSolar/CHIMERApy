from sunpy.net import Fido, attrs as a
import astropy.units as u
import os
from sunpy.map import Map
import numpy as np
import sys
import chimerapy.chimera as chimera_new
import matplotlib.pyplot as plt
import chimerapy.chimera_original as chimera_old

"""
date = "2016/11/04"
wavelength = 193 * u.Angstrom

script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "downloaded_data")
os.makedirs(save_path, exist_ok=True)

results = Fido.search(
    a.Time(f"{date} 00:00:00", f"{date} 00:00:30", near=True),
    a.Instrument("AIA"),
    a.Wavelength(wavelength)
)

if results:
        downloaded_files = Fido.fetch(results[0], path=save_path)
        if downloaded_files:
            print(f"Downloaded file: {downloaded_files[0]}")
        else:
            print("Download failed.")
else:
    print("No files found for the specified date and wavelength.")
"""

path_171 = r"downloaded_data/AIA20161031_1104_0171.fits"
path_193 = r"downloaded_data/AIA20161031_1104_0193.fits"
path_211 = r"downloaded_data/AIA20161031_1104_0211.fits"

m171 = Map(path_171)
m193 = Map(path_193)
m211 = Map(path_211)
imhmi = "https://solarmonitor.org/data/2016/09/22/fits/shmi/shmi_maglc_fd_20160922_094640.fts.gz"

labeled_mask, ch_mask, coronal_holes = chimera_new.chimera(m171, m193, m211)

if coronal_holes:
    print("These are the new CHIMERA results")
else:
    print("No coronal holes found for new CHIMERA.")

circ, data, datb, datc, dattoarc, hedb, iarr, props, rs, slate, center, xgrid, ygrid = chimera_old.chimera([path_171], [path_193], [path_211], [imhmi])

print("Chimera_original processing complete.")

num_coronal_holes = 0
for i in range(1, props.shape[1]):
    if props[0, i].isdigit():
        num_coronal_holes += 1
print(f"Chimera_original found {num_coronal_holes} coronal holes.")

for i in range(1, num_coronal_holes + 1):
    print(f"Coronal Hole {i}:")
    print(f"  ID: {props[0, i]}")
    print(f"  XCEN: {props[1, i]}")
    print(f"  YCEN: {props[2, i]}")
    print(f"  CENTROID: {props[3, i]}")
    print(f"  X_EB: {props[4, i]}")
    print(f"  Y_EB: {props[5, i]}")
    print(f"  X_WB: {props[6, i]}")
    print(f"  Y_WB: {props[7, i]}")
    print(f"  X_NB: {props[8, i]}")
    print(f"  Y_NB: {props[9, i]}")
    print(f"  X_SB: {props[10, i]}")
    print(f"  Y_SB: {props[11, i]}")
    print(f"  WIDTH: {props[12, i]}")
    print(f"  WIDTH: {props[13, i]}")
    print(f"  AREA: {props[14, i]}")
    print(f"  AREA%: {props[15, i]}")
    print(f"  <B>: {props[16, i]}")
    print(f"  <B+>: {props[17, i]}")
    print(f"  <B->: {props[18, i]}")
    print(f"  BMAX: {props[19, i]}")
    print(f"  BMIN: {props[20, i]}")
    print(f"  TOT_B+: {props[21, i]}")
    print(f"  TOT_B-: {props[22, i]}")
    print(f"  <PHI>: {props[23, i]}")
    print(f"  <PHI+>: {props[24, i]}")
    print(f"  <PHI->: {props[25, i]}")

plt.figure()
plt.imshow(iarr.reshape(1024, 4, 1024, 4).sum(axis=(1, 3)))
plt.title(f"Old Coronal Hole Mask")
plt.close()

plt.figure()
plt.imshow(ch_mask)
plt.title(f"New Coronal Hole Mask")
plt.close()

plt.figure()
plt.imshow(m193.data, vmin=50, vmax=500)
plt.title(f"m193 image")
plt.close()

plt.figure()
plt.contour(iarr.reshape(1024, 4, 1024, 4).sum(axis=(1, 3)), colors='w', linewidths=0.5)
plt.imshow(m193.data, vmin=50, vmax=500)
plt.title(f"Old Coronal Hole Overlay")
plt.close()

plt.figure()
plt.contour(ch_mask, colors='w', linewidths=0.5)
plt.imshow(m193.data, vmin=50, vmax=500)
plt.title(f"New Coronal Hole Overlay")
plt.close()