import numpy as np
from scipy import ndimage
import sys

class ClumpFind:
    """
    Custom implementation of Clumpfind (Williams et al. 1994).
    """
    def __init__(self, data, wcs=None):
        self.data = data
        self.wcs = wcs
        self.clump_mask = np.zeros_like(self.data, dtype=int)
        self.clump_id_counter = 0

    def run(self, min_val, step, min_pix=5):
        """
        Run the Clumpfind algorithm.

        Args:
            min_val (float): Integration or Intensity threshold to start (lowest level).
            step (float): Step size for contour levels.
            min_pix (int): Minimum number of pixels for a valid clump.

        Returns:
            np.ndarray: Label mask where each clump has a unique integer ID.
        """
        # 1. Define Contour Levels
        # Generate levels from max down to min_val
        max_val = np.nanmax(self.data)

        if np.isnan(max_val) or max_val < min_val:
            print(f"Invalid max_val: {max_val}. Lower than min_val {min_val} or NaN.")
            return np.zeros_like(self.data, dtype=int)

        # Safety Check for array size
        # If step is too small, levels array explodes
        range_val = max_val - min_val
        n_steps = range_val / step
        if n_steps > 10000:
            print(f"Warning: Too many steps ({n_steps}). Clamping to 1000 steps.")
            step = range_val / 1000.0

        levels = np.arange(min_val, max_val + step, step)
        levels = levels[::-1] # Check from highest to lowest

        self.clump_mask = np.zeros_like(self.data, dtype=int)
        self.clump_id_counter = 0
        
        n_levels = len(levels)
        print(f"ClumpFind: Processing {n_levels} contour levels...")

        # 2. Iterate through levels
        for i, level in enumerate(levels):
            # Progress update
            if i % 5 == 0 or i == n_levels - 1:
                progress = (i + 1) / n_levels * 100
                sys.stdout.write(f"\rClumpFind Progress: {progress:.1f}% (Level {i+1}/{n_levels})")
                sys.stdout.flush()

            # Create a binary mask for the current level
            binary_mask = self.data > level

            # Label connected components at this level
            # structure=np.ones((3,3)) defines connectivity (diagonals included for 2D)
            # For Clumpfind, usually we want strict connectivity (4 for 2D, 6 for 3D) to keep peaks separate
            # so we use generate_binary_structure(ndim, 1)
            ndim = self.data.ndim
            structure = ndimage.generate_binary_structure(ndim, 1) # 1-connectivity (cross)

            # Regions above current level
            labeled_regs, num_regs = ndimage.label(binary_mask, structure=structure)

            if num_regs == 0:
                continue

            # Analyze each connected region at this level
            slices = ndimage.find_objects(labeled_regs)

            for i in range(num_regs):
                label = i + 1
                region_slice = slices[i]

                # Extract the mask for this specific region
                # This is a boolean mask of the current region within its bounding box (roughly)
                # But to access data we should use the full mask or careful indexing
                # Let's interact with the full clump_mask

                # Extract sub-arrays for faster processing
                # We work within the bounding box (region_slice) to avoid full-size mask operations
                sub_labeled = labeled_regs[region_slice]
                sub_region_mask = (sub_labeled == label)
                sub_clump_mask = self.clump_mask[region_slice]

                # Find IDs of existing clumps that overlap with this new region (within the slice)
                existing_labels = sub_clump_mask[sub_region_mask]
                existing_ids = np.unique(existing_labels)
                existing_ids = existing_ids[existing_ids > 0] # Ignore 0

                if len(existing_ids) == 0:
                    # Case 1: No overlap -> New Clump
                    self.clump_id_counter += 1
                    # Update only within the slice using the sub_region_mask
                    # sub_clump_mask is a view, so this updates the main array
                    sub_clump_mask[sub_region_mask] = self.clump_id_counter

                elif len(existing_ids) == 1:
                    # Case 2: One overlap -> Extension
                    clump_id = existing_ids[0]
                    sub_clump_mask[sub_region_mask] = clump_id

                else:
                    # Case 3: Merge (Friends-of-Friends)
                    # We grow existing labels into unassigned pixels

                    # Unassigned mask WITHIN THE SLICE
                    sub_unassigned = sub_region_mask & (sub_clump_mask == 0)
                    if not np.any(sub_unassigned):
                        continue

                    # We need a working copy of labels in this slice
                    current_labels = sub_clump_mask.copy()

                    # We only care about growing into 'sub_region_mask'
                    # Iterative dilation
                    # For strict connectivity:
                    structure = ndimage.generate_binary_structure(ndim, 1)

                    # Mask of pixels we want to fill (valid region but currently 0)
                    # This is essentially sub_unassigned, but let's be explicit
                    target_mask = sub_unassigned & (current_labels == 0)

                    while np.any(target_mask):
                        # Dilate the current non-zero labels
                        # We use grey_dilation or maximum_filter to propagate labels?
                        # maximum_filter propagates the largest label. This might bias towards larger IDs if they touch?
                        # Williams: "pixels are added to the clumps they touch".
                        # If a pixel touches label 1 and label 2? Conflict handling.
                        # Standard FoF usually creates a Voronoi-like partition.
                        # maximum_filter is a reasonable approximation for simultaneous growth.

                        dilated = ndimage.maximum_filter(current_labels, footprint=structure)

                        # Identify new pixels to fill:
                        # Must be in target_mask AND have been touched by dilation (dilated > 0)
                        newly_filled = (dilated > 0) & target_mask

                        if not np.any(newly_filled):
                            # No progress - maybe disconnected components in target_mask that can't be reached?
                            # This shouldn't happen if region is connected, unless seeds are disjoint from targets within this region?
                            # But 'region' comes from label(), so it is connected.
                            # 'seeds' and 'targets' are both parts of this connected region.
                            break

                        # Update labels
                        current_labels[newly_filled] = dilated[newly_filled]

                        # Update target mask (remove filled pixels)
                        target_mask[newly_filled] = False

                    # Write back to main mask
                    # Only update the pixels that were originally unassigned
                    # current_labels now has the filled values

                    # The 'unassigned_mask' corresponds to where we wanted to fill.
                    # We can just write back the changed parts.

                    # sub_unassigned is a boolean mask of the SLICE
                    # current_labels corresponds to the SLICE

                    # We need to act on the main self.clump_mask using the SLICE + MASK

                    # Get the specific pixels in the slice that were updated
                    update_values = current_labels[sub_unassigned]

                    # Apply to main mask
                    # We need to access self.clump_mask[region_slice] and update specific pixels
                    # self.clump_mask[region_slice] is a view?
                    # Yes, basic slicing returns a view.

                    # However, applying a boolean mask 'sub_unassigned' to that view might trigger a copy if not done carefully.
                    # self.clump_mask[region_slice][sub_unassigned] = ... works if __setitem__ handles it.

                    # Safer way:
                    slice_view = self.clump_mask[region_slice]
                    slice_view[sub_unassigned] = update_values


        # 3. Post-processing
        # Remove small clumps (min_pix)
        if min_pix > 0:
            print(f"ClumpFind: Filtering small clumps (<{min_pix} pix)...")
            # Vectorized size filtering
            # 1. Count pixels per label
            # bincount works on 1D array of non-negative integers
            counts = np.bincount(self.clump_mask.ravel())
            
            # 2. Identify dense labels (size >= min_pix)
            # Create a mapping array: existing_label -> new_label (or 0 if filtered)
            # For ClumpFind, we just want to zero out small ones, preserving other IDs?
            # Or do we want to be safe? The original code just zeroed them out.
            
            # map_array[label] = label if size >= min_pix else 0
            map_array = np.arange(len(counts), dtype=self.clump_mask.dtype)
            map_array[counts < min_pix] = 0 # Zero out small clumps
            
            # 3. Apply mapping
            self.clump_mask = map_array[self.clump_mask]

        print("\nClumpFind: Finished.")
        return self.clump_mask

    def get_catalog(self):
        """
        Generate a catalog of properties for detected clumps.
        Requires self.clump_mask to be populated.
        """
        if self.clump_mask is None:
            return []

        labels = np.unique(self.clump_mask)
        labels = labels[labels > 0]

        from takefits.logic.cloud_catalog_utils import calculate_moments_and_props

        catalog = []
        for l in labels:
            props = calculate_moments_and_props(self.data, self.clump_mask, l, self.wcs)
            if props:
                catalog.append(props)

        return catalog
