import numpy as np
from sunpy.map import all_coordinates_from_map, coordinate_is_on_solar_disk, Map
from skimage import measure
import matplotlib.pyplot as plt
import astropy.units as u
from sunpy.coordinates import SphericalScreen

m171 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits")
m193 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits")
m211 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits")

def generate_candidate_mask(m171, m193, m211):
    r"""
    Generate Chimera mask.

    Parameters
    ----------
    m171 : `sunpy.map.Map
        This is the 171 Ångström UV map.
    m193 : `sunpy.map.Map
        This is the 193 Ångström UV map.
    m211 : `sunpy.map.Map
        This is the 211 Ångström UV map.

    Returns
    -------
    numpy.ndarray
        A binary version the Chimera mask.
    """

    coords = all_coordinates_from_map(m171)
    disk_mask = coordinate_is_on_solar_disk(coords)

    m171 = m171 / m171.exposure_time
    m193 = m193 / m193.exposure_time
    m211 = m211 / m211.exposure_time

    threshold_171v193 = 1.5102
    threshold_171v211 = 0.6357
    threshold_193v211 = 0.7

    d171_min, d171_max = 1.2, 3.9
    d193_min, d193_max = 1.4, 3.0
    d211_min, d211_max = 0.8, 2.7

    d171_clipped = np.clip(np.log10(m171.data), d171_min, d171_max)
    d171_clipped_scaled = ((d171_clipped - d171_min) / (d171_max - d171_min)) * 255

    d193_clipped = np.clip(np.log10(m193.data), d193_min, d193_max)
    d193_clipped_scaled = ((d193_clipped - d193_min) / (d193_max - d193_min)) * 255

    d211_clipped = np.clip(np.log10(m211.data), d211_min, d211_max)
    d211_clipped_scaled = ((d211_clipped - d211_min) / (d211_max - d211_min)) * 255

    mask_171_211 = (d171_clipped_scaled / d211_clipped_scaled) >= (
        (np.mean(m171.data[disk_mask]) * threshold_171v211) / np.mean(m211.data[disk_mask])
    )

    mask_211_193 = (d211_clipped_scaled + d193_clipped_scaled) < (
        threshold_193v211 * (np.mean(m193.data[disk_mask]) + np.mean(m211.data[disk_mask]))
    )

    mask_171_193 = (d171_clipped_scaled / d193_clipped_scaled) >= (
        (np.mean(m171.data[disk_mask]) * threshold_171v193) / np.mean(m193.data[disk_mask])
    )

    final_mask = mask_171_211 * mask_211_193 * mask_171_193
    return final_mask

def filter_by_area(mask, min_area=100):
    """
    Filters the CH mask to keep only coronal holes that have an area greater than or equal to "min_area".
    """
    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask)

    filtered_mask = np.zeros_like(mask)
    for region in regions:
        if region.area >= min_area:
            filtered_mask[labeled_mask == region.label] = 1

    return filtered_mask

def get_coronal_holes(ch_mask, map_obj):
    """
    Extract coronal hole properties from the binary mask.

    Parameters      
    ----------
    ch_mask : `numpy.ndarray
        The binary mask of coronal holes.
    map_obj : `sunpy.map.Map
        The map object from which the mask was generated.

    Returns
    -------
    list of dict
        A list of dictionaries containing properties of each coronal hole.
    """
    coronal_holes = []
    labeled_mask = measure.label(ch_mask)

    solar_radius_arcsec = map_obj.rsun_obs
    pixel_scale_arcsec = map_obj.scale[0]
    solar_radius_pixels = (solar_radius_arcsec / pixel_scale_arcsec).value
    solar_disk_area_pixels = np.pi * (solar_radius_pixels ** 2)

    for region in measure.regionprops(labeled_mask):
        min_row, min_col, max_row, max_col = region.bbox
        width_pixels = {
            "E-W": max_col - min_col,
            "N-S": max_row - min_row
        }
        
        with SphericalScreen(map_obj.observer_coordinate):
            bottom_left = map_obj.pixel_to_world(min_col * u.pix, min_row * u.pix).transform_to("heliographic_stonyhurst")
            top_right = map_obj.pixel_to_world(max_col * u.pix, max_row * u.pix).transform_to("heliographic_stonyhurst")

            width_heliographic = {
                "E-W": abs(top_right.lon.deg - bottom_left.lon.deg),
                "N-S": abs(top_right.lat.deg - bottom_left.lat.deg)
            }


        centroid = region.centroid
        centroid_world = map_obj.pixel_to_world(centroid[1] * u.pix, centroid[0] * u.pix)
        area_percentage = (region.area / solar_disk_area_pixels) * 100
        coronal_holes.append({
            "area": region.area,
            "area_percentage": area_percentage,
            "centroid": centroid_world,
            "width_pixels": width_pixels,
            "width_heliographic": width_heliographic
        })

        coronal_holes = sorted(coronal_holes, key=lambda ch: ch["area"], reverse=True)
        for idx, ch in enumerate(coronal_holes, start=1):
            ch["id"] = idx

    return coronal_holes

ch_mask = generate_candidate_mask(m171, m193, m211)
filtered_ch_mask = filter_by_area(ch_mask, min_area=100)
coronal_holes = get_coronal_holes(filtered_ch_mask, m171)

for ch in coronal_holes:
    print(
        f"CH {ch['id']}: "
        f"Area = {ch['area_percentage']:.2f}% of the solar disk, "
        f"Centroid in arcseconds = {ch['centroid'].Tx.value:.2f}, {ch['centroid'].Ty.value:.2f}, "
        f"E-W Width = {ch['width_heliographic']['E-W']:.2f} °, "
        f"N-S Width = {ch['width_heliographic']['N-S']:.2f} °"
    )
