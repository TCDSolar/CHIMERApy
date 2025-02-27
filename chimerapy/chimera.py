import numpy as np
from sunpy.map import all_coordinates_from_map, coordinate_is_on_solar_disk


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

    threshold_171v193 = 0.6357
    threshold_171v211 = 0.7
    threshold_193v211 = 1.5102

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
