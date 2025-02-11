import unittest
import numpy as np
from sunpy.map import Map

# Assuming process_solar_images and plot_mask have been imported from your module
from chimerapy.chimera_func import process_solar_images

class TestSolarImageProcessing(unittest.TestCase):
    r"""
    Unit tests for the `process_solar_images` function.

    This class tests the functionality of the `process_solar_images` function.
    It ensures that the function returns a binary mask with the expected shape
    and values based on the input AIA solar images.

    Parameters
    ----------
    m171 : `sunpy.map.Map
        The 171 Ångström UV solar image.
    m193 : `sunpy.map.Map
        The 193 Ångström UV solar image.
    m211 : `sunpy.map.Map
        The 211 Ångström UV solar image.
    """
    def setUp(self):
        r"""
        Set up the test environment by loading the AIA solar images.

        This method loads the AIA solar images (171 Ångström, 193 Ångström,
        and 211 Ångström) from the Stanford JSOC URLs.
        """
        self.m171 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/09/22/H1200/AIA20160922_1200_0171.fits")
        self.m193 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/09/22/H1200/AIA20160922_1200_0193.fits")
        self.m211 = Map("https://jsoc1.stanford.edu/data/aia/synoptic/2016/09/22/H1200/AIA20160922_1200_0211.fits")
        
    
    def test_process_solar_images(self):
        r"""
        Test the function `process_solar_images` .

        This test method tests that `process_solar_images`
        creates a binary mask that fits the following rules:
        
        - The result is a numpy array.
        - The shape of the mask matches the input image data shape.
        - The mask only contains binary values (0 or 1).

        Raises
        ------
        AssertionError
            If any of the conditions (numpy array, shape match, binary values) are not met.
        """
        final_mask = process_solar_images(self.m171, self.m193, self.m211)
        
        self.assertIsInstance(final_mask, np.ndarray)
        
        self.assertEqual(final_mask.shape, self.m171.data.shape)
        
        unique_values = np.unique(final_mask)
        self.assertTrue(np.all(np.isin(unique_values, [0, 1])))

if __name__ == "__main__":
    unittest.main()