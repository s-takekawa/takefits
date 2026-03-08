import numpy as np
from skimage.segmentation import watershed
from skimage.morphology import h_maxima
from skimage.measure import label

class FellWalker:
    def __init__(self, data, wcs=None):
        self.data = data
        self.wcs = wcs
        self.mask = None

    def run(self, min_val, min_dip, min_pix):
        """
        Run the FellWalker-like (Watershed) algorithm.

        Parameters:
        -----------
        min_val : float
            Minimum intensity value to be considered (threshold).
        min_dip : float
            Prominence threshold. Peaks must stand out by at least this amount
            to be considered separate markers.
        min_pix : int
            Minimum number of pixels for a clump to be valid.

        Returns:
        --------
        mask : np.ndarray
            Integer mask where each clump has a unique ID.
        """
        # 1. Thresholding
        print("FellWalker: Starting...")
        # We only care about regions > min_val
        valid_mask = self.data > min_val

        # If no data above threshold, return empty
        if not np.any(valid_mask):
            print("FellWalker: No data above threshold.")
            return np.zeros_like(self.data, dtype=np.int32)

        # 2. Prepare Topography for Watershed
        # Watershed works on "basins" (minima).
        # So we invert the image: inverted = max - data
        inverted = np.max(self.data) - self.data

        # 3. Find Markers (H-Maxima)
        # We want to identify peaks that are significant.
        print("FellWalker: Finding peaks (h_maxima)...")
        
        # IMPORTANT: h_maxima works on the original (non-inverted) data to find peaks.
        # Replace NaNs with min_val or lower to avoid issues?
        data_clean = np.nan_to_num(self.data, nan=np.nanmin(self.data))

        # Find local maxima with prominence >= min_dip
        peaks_binary = h_maxima(data_clean, min_dip)

        # Filter peaks that are below min_val
        peaks_binary = peaks_binary & (self.data > min_val)

        # If no peaks found with h_maxima (e.g. if min_dip is too high),
        # fallback to simple local max or return empty?
        if not np.any(peaks_binary):
             # Fallback: maybe just simple local max?
             # For now, return empty as it implies no significant structures.
             print("FellWalker: No peaks found.")
             return np.zeros_like(self.data, dtype=np.int32)

        # Label the markers
        markers, num_markers = label(peaks_binary, return_num=True)

        # 4. Run Watershed
        # mask argument restricts the filling to valid_mask
        print(f"FellWalker: Found {num_markers} markers. Running watershed...")
        labels = watershed(inverted, markers, mask=valid_mask)

        # 5. Filter by Size (min_pix)
        # Remove small clumps
        print(f"FellWalker: Filtering clumps smaller than {min_pix} pixels...")
        # Vectorized size filtering and re-labeling
        # 1. Count pixels per label
        counts = np.bincount(labels.ravel())
        
        # 2. Create mapping array
        # map_array[old_label] = new_label
        map_array = np.zeros(len(counts), dtype=np.int32)
        
        # Identify valid labels (size >= min_pix)
        valid_mask = (counts >= min_pix)
        valid_mask[0] = False # Ensure background (0) stays 0
        
        # Assign new continuous IDs to valid labels
        num_valid = np.sum(valid_mask)
        map_array[valid_mask] = np.arange(1, num_valid + 1, dtype=np.int32)
        
        # 3. Apply mapping to generate final mask
        final_mask = map_array[labels]
        
        # Calculate number of clumps for reporting
        new_id_count = num_valid

        print(f"FellWalker: Finished. Identified {new_id_count} clumps.")
        self.mask = final_mask
        return final_mask

    def get_catalog(self):
        """
        Generate a catalog of properties for the identified clumps.
        """
        if self.mask is None:
            return []

        from takefits.logic.cloud_catalog_utils import calculate_moments_and_props

        # Get all unique labels
        labels = np.unique(self.mask)
        # Remove background (0)
        labels = labels[labels > 0]

        catalog = []

        for l in labels:
            props = calculate_moments_and_props(self.data, self.mask, l, self.wcs)
            if props:
                catalog.append(props)

        return catalog
