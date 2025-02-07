import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk, pixelate_coord_path, sample_at_coords
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.units import UnitsError
from astropy.visualization import make_lupton_rgb

def process_solar_images(m131, m171, m193, m211, m335, m94):
    # Step 1: Create the figure layout
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

    # Temperature mapping for each wavelength
    temps = {
        "131": 0.4,
        "171": 0.7,
        "193": 1.2,
        "211": 2.0,
        "335": 2.5,
        "94": 6.3,
    }

    # Part 1
    for m in [m131, m171, m193, m211, m335, m94]:
        wave_str = f"{m.wavelength.value:0.0f}"
        line_coords = SkyCoord([-200, 0], [-100, -100], unit=(u.arcsec, u.arcsec), frame=m.coordinate_frame)
        m.plot(axes=axes[wave_str], clip_interval=(0.1, 99.9) * u.percent)
        axes[wave_str].set_title(f"{wave_str} Å {temps[wave_str]} MK")
        axes[wave_str].plot_coord(line_coords, "w", linewidth=1.5)
        axes[wave_str].set_facecolor("k")

    # Part 2
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

    # Part 3
    coords = all_coordinates_from_map(m171)
    disk_mask = coordinate_is_on_solar_disk(coords)

    m171 = m171 / m171.exposure_time
    m193 = m193 / m193.exposure_time
    m211 = m211 / m211.exposure_time

    # Part 4
    threshold_171v193 = 0.6357
    threshold_171v211 = 0.7
    threshold_193v211 = 1.5102
    xx = np.linspace(0, 300, 500)
    fit171v193 = 2.25 * xx**threshold_171v193
    fit171v211 = 0.5 * xx**threshold_171v211
    fit193v211 = 2600 * xx**-threshold_193v211

    # Part 5
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

    # Part 6
    mask_171_211 = (d171_clipped_scaled / d211_clipped_scaled) >= (
        (np.mean(m171.data[disk_mask]) * threshold_171v193) / np.mean(m211.data[disk_mask])
    )

    mask_211_193 = (d211_clipped_scaled + d193_clipped_scaled) < (
        0.7 * (np.mean(m193.data[disk_mask]) + np.mean(m211.data[disk_mask]))
    )

    mask_171_193 = (d171_clipped_scaled / d193_clipped_scaled) >= (
        (np.mean(m171.data[disk_mask]) * threshold_193v211) / np.mean(m193.data[disk_mask])
    )

    # Part 7
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

        for contour in contours[:6]:
            axes["seg"].plot_coord(contour, color="w", linewidth=0.5)
        
        plt.show()

    final_mask(m171, tri_color_img, mask_171_211, mask_211_193, mask_171_193)

process_solar_images(m131, m171, m193, m211, m335, m94)