import numpy as np
from sunpy.map import all_coordinates_from_map, coordinate_is_on_solar_disk, Map
from skimage import measure
import matplotlib.pyplot as plt
import astropy.units as u
from sunpy.coordinates import SphericalScreen
from matplotlib import colors

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

def calculate_cosine_correction(im_map: Map):
    """
    Find the cosine correction values for on-disk pixels.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Processed SunPy map.

    Returns
    -------
    cos_correction : `~numpy.ndarray`
        Array of cosine correction factors for each pixel. Values greater than a threshold (edge) are set to 1.
    """
    coordinates = all_coordinates_from_map(im_map)
    on_disk = coordinate_is_on_solar_disk(coordinates)

    cos_correction = np.ones_like(im_map.data)

    radial_angle = np.arccos(np.cos(coordinates.Tx[on_disk]) * np.cos(coordinates.Ty[on_disk]))
    cos_cor_ratio = (radial_angle / im_map.rsun_obs).value
    cos_cor_ratio = np.clip(cos_cor_ratio, -1, 1)

    cos_correction[on_disk] = 1 / (np.cos(np.arcsin(cos_cor_ratio)))

    return cos_correction

@u.quantity_input(min_area=u.m**2)
def filter_by_area(mask, map_obj, solar_radius_pixels, min_area=1e10 * u.m**2):
    solar_radius_meters = map_obj.rsun_meters
    pixel_scale_arcsec = map_obj.scale[0].value
    pixel_scale_meters = (pixel_scale_arcsec * u.arcsec).to(u.rad).value * solar_radius_meters
    pixel_area_meters2 = pixel_scale_meters**2
    min_area_pixels = (min_area / pixel_area_meters2).decompose().value

    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask)

    for region in regions:
        if region.area < min_area_pixels:
            labeled_mask[labeled_mask == region.label] = 0

    filtered_regions = measure.regionprops(labeled_mask)
    return labeled_mask, filtered_regions

def get_coronal_holes(filtered_regions, map_obj):
    coronal_holes = []

    for region in filtered_regions:
        coords = region.coords
        world_coords = map_obj.pixel_to_world(coords[:, 1] * u.pix, coords[:, 0] * u.pix)
        heliographic_coords = world_coords.transform_to("heliographic_stonyhurst")

        min_lon = heliographic_coords.lon.min().deg
        max_lon = heliographic_coords.lon.max().deg
        min_lat = heliographic_coords.lat.min().deg
        max_lat = heliographic_coords.lat.max().deg

        extent_lon = max_lon - min_lon
        extent_lat = max_lat - min_lat

        centroid = region.centroid
        centroid_world = map_obj.pixel_to_world(centroid[1] * u.pix, centroid[0] * u.pix)

        pixel_scale_arcsec = map_obj.scale[0].value
        pixel_scale_meters = (pixel_scale_arcsec * u.arcsec).to(u.rad).value * map_obj.rsun_meters
        pixel_area_meters2 = pixel_scale_meters**2

        cos_theta = np.abs(np.cos(heliographic_coords.lat.rad) * np.cos(heliographic_coords.lon.rad))
        adjusted_pixel_area_meters2 = pixel_area_meters2 / cos_theta
        adjusted_pixel_area_meters2 = np.nan_to_num(adjusted_pixel_area_meters2)

        area_meters2 = np.sum(adjusted_pixel_area_meters2)

        coronal_holes.append({
            "area_pixels": region.area,
            "area_meters2": area_meters2,
            "centroid_world": centroid_world,
            "centroid_heliographic": centroid_world.transform_to("heliographic_stonyhurst"),
            "min_lon": min_lon,
            "max_lon": max_lon,
            "min_lat": min_lat,
            "max_lat": max_lat,
            "extent_lon": extent_lon,
            "extent_lat": extent_lat,
        })

    coronal_holes = sorted(coronal_holes, key=lambda ch: ch["area_pixels"], reverse=True)
    for idx, ch in enumerate(coronal_holes, start=1):
        ch["id"] = idx

    return coronal_holes

def map_threshold(im_map):
    """
    Set off-disk pixels to black and clip the vmin and vmax of the map.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Unprocessed magnetogram map.

    Returns
    -------
    im_map : `~sunpy.map.Map`
        Processed magnetogram map.
    """
    im_map.data[~coordinate_is_on_solar_disk(all_coordinates_from_map(im_map))] = np.nan
    im_map.cmap.set_bad("k")
    im_map.plot_settings["norm"] = colors.Normalize(vmin=-200, vmax=200)
    return im_map

m171 = map_threshold(m171)
m193 = map_threshold(m193)
m211 = map_threshold(m211)

cos_correction_171 = calculate_cosine_correction(m171)
cos_correction_193 = calculate_cosine_correction(m193)
cos_correction_211 = calculate_cosine_correction(m211)

m171_corrected = m171.data * cos_correction_171
m193_corrected = m193.data * cos_correction_193
m211_corrected = m211.data * cos_correction_211

m171 = Map(m171_corrected, m171.meta)
m193 = Map(m193_corrected, m193.meta)
m211 = Map(m211_corrected, m211.meta)

solar_radius_arcsec = m171.rsun_obs
pixel_scale_arcsec = m171.scale[0]
solar_radius_pixels = (solar_radius_arcsec / pixel_scale_arcsec).value

ch_mask = generate_candidate_mask(m171, m193, m211)
labeled_mask, filtered_regions = filter_by_area(ch_mask, m171, solar_radius_pixels, min_area=1e10 * u.m**2)

coronal_holes = get_coronal_holes(filtered_regions, m171)

for ch in coronal_holes:
    print(
        f"CH {ch['id']}: "
        f"Area = {ch['area_meters2']:.2e}, "
        f"Centroid in arcseconds = {ch['centroid_world'].Tx.value:.2f}, {ch['centroid_world'].Ty.value:.2f}, "
        f"E-W Extent = {ch['extent_lon']:.2f} °, "
        f"N-S Extent = {ch['extent_lat']:.2f} °"
    )
