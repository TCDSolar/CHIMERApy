import numpy as np
from matplotlib import colors
from numpy.typing import NDArray
from skimage import measure
from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk

import astropy.units as u
from astropy.units import Quantity


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


def get_area_map(map):
    return np.array(1) * u.m**2


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

    radial_angle = np.arccos(np.cos(coordinates.Tx) * np.cos(coordinates.Ty))
    cos_cor_ratio = (radial_angle / im_map.rsun_obs).decompose()
    cos_cor_ratio = np.clip(cos_cor_ratio, -1, 1)  # Clip to on disk values

    cos_correction = 1 / (np.cos(np.arcsin(cos_cor_ratio)))

    return cos_correction, on_disk


@u.quantity_input()
def filter_by_area(mask: NDArray, map_obj: Map, min_area: Quantity["area"] = 1e10 * u.m**2):  # noqa: F821
    solar_radius = map_obj.rsun_meters
    pixel_scale = map_obj.scale[0] * 1 * u.pixel
    # Sun center approx flat over 1 pixel
    pixel_size = pixel_scale.to_value(u.rad) * solar_radius
    pixel_area = pixel_size**2

    cos_correction, on_disk = calculate_cosine_correction(map_obj)
    area_map = pixel_area / cos_correction

    labeled_mask = measure.label(mask * on_disk)
    regions = measure.regionprops(labeled_mask)

    filtered_regions = []
    for region in regions:
        region_mask = labeled_mask == region.label
        region_surface_area = area_map[region_mask].sum()
        if region_surface_area >= min_area and not np.all(region_mask & on_disk):
            region.surface_area = region_surface_area
            filtered_regions.append(region)
        else:
            labeled_mask[region_mask] = 0

    return labeled_mask, filtered_regions


def get_coronal_holes(filtered_regions, map_obj, labeled_mask):
    coronal_holes = []

    for region in filtered_regions:
        coords = region.coords
        world_coords = map_obj.pixel_to_world(coords[:, 1] * u.pix, coords[:, 0] * u.pix)
        heliographic_coords = world_coords.transform_to("heliographic_stonyhurst")

        min_lon = heliographic_coords.lon.min()
        max_lon = heliographic_coords.lon.max()
        min_lat = heliographic_coords.lat.min()
        max_lat = heliographic_coords.lat.max()

        extent_lon = max_lon - min_lon
        extent_lat = max_lat - min_lat

        centroid = region.centroid
        centroid_world = map_obj.pixel_to_world(centroid[1] * u.pix, centroid[0] * u.pix)

        # breakpoint()

        coronal_holes.append(
            {
                "area_pixels": region.area,
                "area_meters2": region.surface_area,
                "centroid_world": centroid_world,
                "centroid_heliographic": centroid_world.transform_to("heliographic_stonyhurst"),
                "min_lon": min_lon,
                "max_lon": max_lon,
                "min_lat": min_lat,
                "max_lat": max_lat,
                "extent_lon": extent_lon,
                "extent_lat": extent_lat,
            }
        )

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


def chimera(m171, m193, m211):
    ch_mask = generate_candidate_mask(m171, m193, m211)
    labeled_mask, filtered_regions = filter_by_area(ch_mask, m171, min_area=1e10 * u.m**2)

    coronal_holes = get_coronal_holes(filtered_regions, m171, labeled_mask)

    for ch in coronal_holes:
        print(
            f"CH {ch['id']}: "
            f"Area = {ch['area_meters2']:.2e}, "
            f"Centroid in arcseconds = {ch['centroid_world'].Tx.value:.2f}, {ch['centroid_world'].Ty.value:.2f}, "
            f"E-W Extent = {ch['extent_lon']:.2f} °, "
            f"N-S Extent = {ch['extent_lat']:.2f} °"
        )


if __name__ == "__main__":
    m171 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits")
    m193 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits")
    m211 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits")
    chimera(m171, m193, m211)
