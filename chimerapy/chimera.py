import numpy as np
from matplotlib import colors
from numpy.typing import NDArray
from skimage import measure
from skimage.draw import polygon2mask
from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk

import astropy.units as u
from astropy.units import Quantity

from chimerapy import log


def generate_candidate_mask(m171, m193, m211):
    r"""
    Generate coronal hole candidate mask based image ratios.

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

    with np.errstate(divide="ignore", invalid="ignore"):
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


def calculate_area_map(im_map: Map):
    """
    Generate map where each pixel is the area the pixel subtends on the solar surface.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Processed SunPy map.

    Returns
    -------
    area_map : `~numpy.ndarray`
        Surface area each pixel subtends on the sun.
    """
    coordinates = all_coordinates_from_map(im_map)
    disk_mask = coordinate_is_on_solar_disk(coordinates)

    pixel_scale = im_map.scale[0] * 1 * u.pixel
    pixel_size = np.arcsin(pixel_scale / im_map.rsun_obs).to_value(u.rad) * im_map.rsun_meters
    pixel_area = pixel_size**2

    radial_angle = np.arccos(np.cos(coordinates.Tx[disk_mask]) * np.cos(coordinates.Ty[disk_mask]))
    ratio = (radial_angle / im_map.rsun_obs).decompose()
    theta = np.arcsin(ratio)
    cos_correction = np.cos(theta)

    area_map = np.full(im_map.data.shape, 0) << pixel_area.unit
    area_map[disk_mask] = pixel_area / cos_correction
    return area_map, disk_mask


@u.quantity_input()
def filter_ch(mask: NDArray, map_obj: Map, min_area: Quantity["area"] = 1e4 * u.Mm**2, on_disk: bool = True):  # noqa: F821
    r"""
    Filter coronal hole candidate masks

    Parameters
    ----------
    mask :
        Candidate CH mask
    map_obj

    min_area
        Remove CH with area below this value.
    on_disk :
        Remove CHs that are not on the disk (above the limb)
    """
    area_map, disk_mask = calculate_area_map(map_obj)

    labeled_mask = measure.label(mask * disk_mask if on_disk else mask)
    regions = measure.regionprops(labeled_mask)

    filtered_regions = []
    for region in regions:
        region_mask = labeled_mask == region.label
        contours = measure.find_contours(region_mask)
        if contours:
            encompassing_contour = sorted(contours, key=lambda x: x.size, reverse=True)[0]
            filled_region_mask = polygon2mask(region_mask.shape, encompassing_contour)
            region_surface_area = area_map[filled_region_mask].sum()
            labeled_mask[filled_region_mask] = region.label
            if region_surface_area >= min_area and not np.all(region_mask & disk_mask):
                region.surface_area = region_surface_area
                filtered_regions.append(region)
            else:
                labeled_mask[region_mask] = 0

    filtered_regions = sorted(filtered_regions, key=lambda region: region.surface_area, reverse=True)

    return labeled_mask, filtered_regions


def get_coronal_holes(filtered_regions, map_obj, labeled_mask):
    coronal_holes = []

    for region in filtered_regions:
        coords = region.coords
        world_coords = map_obj.pixel_to_world(coords[:, 1] * u.pix, coords[:, 0] * u.pix)

        wb = world_coords[np.nanargmax(world_coords.Tx)]
        eb = world_coords[np.nanargmin(world_coords.Tx)]

        nb = world_coords[np.nanargmax(world_coords.Ty)]
        sb = world_coords[np.nanargmin(world_coords.Ty)]

        extent_lon = (
            wb.transform_to("heliographic_stonyhurst").lon - eb.transform_to("heliographic_stonyhurst").lon
        )
        extent_lat = (
            sb.transform_to("heliographic_stonyhurst").lat - nb.transform_to("heliographic_stonyhurst").lat
        )

        centroid = region.centroid
        centroid_world = map_obj.pixel_to_world(centroid[1] * u.pix, centroid[0] * u.pix)

        coronal_holes.append(
            {
                "area_pixels": region.area,
                "area_meters2": region.surface_area,
                "centroid_world": centroid_world,
                "centroid_heliographic": centroid_world.transform_to("heliographic_stonyhurst"),
                "eb": eb,
                "wb": wb,
                "sb": sb,
                "nb": nb,
                "extent_lon": extent_lon,
                "extent_lat": extent_lat,
            }
        )

    coronal_holes = sorted(coronal_holes, key=lambda ch: ch["area_meters2"], reverse=True)
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
    on_disk = coordinate_is_on_solar_disk(all_coordinates_from_map(im_map))
    im_map.data[~on_disk] = np.nan
    im_map.cmap.set_bad("k")
    im_map.plot_settings["norm"] = colors.Normalize(vmin=-200, vmax=200)
    return im_map


def chimera(m171, m193, m211):
    ch_mask = generate_candidate_mask(m171, m193, m211)
    labeled_mask, filtered_regions = filter_ch(ch_mask, m171)

    coronal_holes = get_coronal_holes(filtered_regions, m171, labeled_mask)

    for ch in coronal_holes:
        print(
            f"CH {ch['id']}: "
            f"Area = {ch['area_meters2'].to('Mm**2'):.2e}, "
            f"Centroid HPC: {ch['centroid_world'].Tx.value:.2f},{ch['centroid_world'].Ty.value:.2f}, "
            f"Centroid HGS: {ch['centroid_heliographic'].lon.value:.2f},{ch['centroid_heliographic'].lat.value:.2f}, "
            f"EB: {ch['eb'].Tx.value:.2f},{ch['eb'].Ty.value:.2f}, "
            f"WB: {ch['wb'].Tx.value:.2f},{ch['wb'].Ty.value:.2f}, "
            f"NB: {ch['nb'].Tx.value:.2f},{ch['nb'].Ty.value:.2f}, "
            f"SB: {ch['sb'].Tx.value:.2f},{ch['sb'].Ty.value:.2f}, "
            f"E: {ch['eb'].transform_to('heliographic_stonyhurst').lon.value:.2f}-W:{ch['wb'].transform_to('heliographic_stonyhurst').lon.value:.2f}, "
            f"E-W Extent = {ch['extent_lon']:.2f} °, "
            f"N: {ch['nb'].transform_to('heliographic_stonyhurst').lat.value:.2f}, S:{ch['sb'].transform_to('heliographic_stonyhurst').lat.value:.2f}"
            f"N-S Extent = {ch['extent_lat']:.2f} °, "
        )
    return ch_mask, labeled_mask, coronal_holes


if __name__ == "__main__":
    m171 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0171.fits")
    m193 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0193.fits")
    m211 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/10/31/H0200/AIA20161031_0232_0211.fits")
    chimera(m171, m193, m211)
