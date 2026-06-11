import numpy as np

def calculate_props_from_indices_values(indices, values, wcs=None, ndim=3, label_id=0):
    """
    Calculate properties from indices and values.
    shared logic for ClumpFind, FellWalker, and Dendrogram.
    """
    if len(values) == 0:
        return None

    # 0. Basic Stats
    npix = len(values)
    flux_sum = np.sum(values)
    peak_val = np.max(values)
    mean_val = np.mean(values)

    # Weights for moments (Intensity)
    weights = values
    sum_w = np.sum(weights)

    if sum_w == 0: sum_w = 1.0 # Avoid division by zero

    # 1. Centroids (Weighted)
    # indices is tuple of arrays 
    # ndim should match len(indices)

    coords_pix = []

    # Calculate simple weighted mean for each axis
    centroids_pix = []
    for i in range(ndim):
        # Ensure indices[i] exists
        if i < len(indices):
            c = np.sum(indices[i] * weights) / sum_w
            centroids_pix.append(c)
        else:
            centroids_pix.append(0.0)

    # Assign to standardized variable names based on dimensionality
    if ndim == 3:
        z_c_pix, y_c_pix, x_c_pix = centroids_pix
    elif ndim == 2:
        y_c_pix, x_c_pix = centroids_pix
        z_c_pix = 0
    else:
        x_c_pix = centroids_pix[0]
        y_c_pix, z_c_pix = 0, 0

    # 2. Second Moments (Dispersion)
    # Sigma^2 = Sum( w * (x - x_c)^2 ) / Sum(w)

    sigmas_pix = []
    
    # Store per-axis sigma for generic access
    sigma_dict = {}

    for i in range(ndim):
        if i < len(indices):
            diff = indices[i] - centroids_pix[i]
            sigma2 = np.sum(weights * (diff**2)) / sum_w
            sigma = np.sqrt(sigma2)
            sigmas_pix.append(sigma)
        else:
            sigmas_pix.append(0.0)

    if ndim == 3:
        z_sig_pix, y_sig_pix, x_sig_pix = sigmas_pix
        sigma_dict['z'] = z_sig_pix
        sigma_dict['y'] = y_sig_pix
        sigma_dict['x'] = x_sig_pix
    elif ndim == 2:
        y_sig_pix, x_sig_pix = sigmas_pix
        z_sig_pix = 0
        sigma_dict['y'] = y_sig_pix
        sigma_dict['x'] = x_sig_pix
    else:
        x_sig_pix = sigmas_pix[0]
        y_sig_pix, z_sig_pix = 0, 0
        sigma_dict['x'] = x_sig_pix

    # 3. Covariance for Ellipse (Spatial: X vs Y)
    # Map indices to X, Y
    # dim 2 (X) corresponds to index -1
    # dim 1 (Y) corresponds to index -2
    
    if ndim >= 2:
        idx_x = indices[-1]
        idx_y = indices[-2]
        
        diff_x = idx_x - x_c_pix
        diff_y = idx_y - y_c_pix

        cov_xy = np.sum(weights * diff_x * diff_y) / sum_w
        var_x = x_sig_pix**2
        var_y = y_sig_pix**2

        term = np.sqrt((var_x - var_y)**2 + 4*cov_xy**2)
        lambda1 = (var_x + var_y + term) / 2
        lambda2 = (var_x + var_y - term) / 2

        maj_sigma_pix = np.sqrt(lambda1)
        min_sigma_pix = np.sqrt(lambda2)
        
        # Standard: CCW from positive X.
        pa_rad = 0.5 * np.arctan2(2*cov_xy, var_x - var_y)
        pa_deg = np.degrees(pa_rad)
    else:
        maj_sigma_pix = 0
        min_sigma_pix = 0
        pa_deg = 0

    # 4. Physical Units (WCS)
    sigma_x_deg = 0
    sigma_y_deg = 0
    sigma_v_kms = 0
    maj_sigma_deg = 0
    min_sigma_deg = 0
    r_eff_deg = 0
    area_deg2 = 0

    glon_deg, glat_deg, velo_kms = 0, 0, 0

    if wcs:
        from astropy.wcs.utils import proj_plane_pixel_scales
        try:
            scales = proj_plane_pixel_scales(wcs.celestial)
            dx_deg = scales[0] # Approx
            dy_deg = scales[1] if len(scales) > 1 else dx_deg

            # Area in deg2
            pixel_area_deg2 = dx_deg * dy_deg
            area_deg2 = npix * pixel_area_deg2

            # R_eff = sqrt(Area / pi)
            r_eff_deg = np.sqrt(area_deg2 / np.pi)

            # Sigmas
            sigma_x_deg = x_sig_pix * dx_deg
            sigma_y_deg = y_sig_pix * dy_deg

            maj_sigma_deg = maj_sigma_pix * max(dx_deg, dy_deg) 
            min_sigma_deg = min_sigma_pix * min(dx_deg, dy_deg)

        except Exception:
            # Fallback
            if hasattr(wcs.wcs, 'cdelt'):
                cdelt = wcs.wcs.cdelt
                dx_deg = abs(cdelt[0])
                dy_deg = abs(cdelt[1])
                area_deg2 = npix * dx_deg * dy_deg
                r_eff_deg = np.sqrt(area_deg2 / np.pi)
                sigma_x_deg = x_sig_pix * dx_deg
                sigma_y_deg = y_sig_pix * dy_deg
                maj_sigma_deg = maj_sigma_pix * dx_deg
                min_sigma_deg = min_sigma_pix * dy_deg

        
        # Velocity & World Coordinate Logic
        try:
            wcs_naxis = wcs.wcs.naxis if hasattr(wcs.wcs, 'naxis') else ndim
            
            # A. Velocity (Manual 3D Calculation)
            # User requested simple calculation from CDELT/CRVAL/CRPIX to avoid unit issues.
            if ndim == 3 and wcs_naxis >= 3:
                 try:
                     spec_axis = 2 # 3rd axis
                     cdelt = wcs.wcs.cdelt[spec_axis]
                     crval = wcs.wcs.crval[spec_axis]
                     crpix = wcs.wcs.crpix[spec_axis] # 1-based
                     
                     # Unit Heuristic: Check CUNIT or CDELT magnitude
                     # If CUNIT is 'm/s', convert to km/s
                     # If CUNIT is missing but CDELT is huge (> 100), assume m/s and convert
                     try:
                         pixel_scale = 1.0
                         cunit = wcs.wcs.cunit[spec_axis].to_string().strip().lower()
                         if cunit == 'm/s':
                             pixel_scale = 0.001
                         elif cunit == '' and abs(cdelt) > 50.0:
                             # Heuristic: Most radio data has channel width < 50 km/s. 
                             # If values are > 50, it is likely m/s (e.g. 3000 m/s).
                             pixel_scale = 0.001
                         
                         cdelt *= pixel_scale
                         crval *= pixel_scale
                     except Exception:
                         # Fallback heuristic
                         if abs(cdelt) > 50.0:
                             cdelt *= 0.001
                             crval *= 0.001

                     # 1. Sigma (Dispersion)
                     sigma_v_kms = z_sig_pix * abs(cdelt)
                     
                     # 2. Centroid
                     # z_c_pix is 0-based index
                     velo_kms = crval + (z_c_pix - (crpix - 1)) * cdelt
                 except Exception:
                     pass

            # B. Spatial World Coordinates (RA/DEC / GLON/GLAT)
            # Use pixel_to_world for spatial axes
            if wcs_naxis == 3:
                # 3D WCS
                try:
                    world = wcs.pixel_to_world(x_c_pix, y_c_pix, z_c_pix)
                    if isinstance(world, (list, tuple)):
                        coord = world[0]
                        # We used manual velocity above, so ignore world[1] or verify
                    else:
                        coord = world
                    
                    if hasattr(coord, 'l'):
                        glon_deg = coord.l.deg
                        glat_deg = coord.b.deg
                    elif hasattr(coord, 'ra'):
                        glon_deg = coord.ra.deg
                        glat_deg = coord.dec.deg
                except Exception:
                    pass
                    
            elif wcs_naxis == 2 or ndim == 2:
                # 2D WCS
                try:
                    try:
                        celestial_wcs = wcs.celestial
                        world = celestial_wcs.pixel_to_world(x_c_pix, y_c_pix)
                    except Exception:
                        world = wcs.pixel_to_world(x_c_pix, y_c_pix)

                    if hasattr(world, 'l'):
                        glon_deg = world.l.deg
                        glat_deg = world.b.deg
                    elif hasattr(world, 'ra'):
                        glon_deg = world.ra.deg
                        glat_deg = world.dec.deg
                except Exception:
                    pass

        except Exception:
            pass

    return {
        'id': int(label_id),
        'peak': peak_val,
        'mean': mean_val,
        'sum': flux_sum,
        'npix': npix,
        'x_pix': x_c_pix,
        'y_pix': y_c_pix,
        'z_pix': z_c_pix,
        'cen_x_deg': glon_deg,
        'cen_y_deg': glat_deg,
        'cen_v_kms': velo_kms,
        'sigma_x_pix': x_sig_pix,
        'sigma_y_pix': y_sig_pix,
        'sigma_z_pix': z_sig_pix,
        'sigma_x_deg': sigma_x_deg,
        'sigma_y_deg': sigma_y_deg,
        'sigma_v_kms': sigma_v_kms,
        'maj_sigma_deg': maj_sigma_deg,
        'min_sigma_deg': min_sigma_deg,
        'pa_deg': pa_deg,
        'area_deg2': area_deg2,
        'r_eff_deg': r_eff_deg
    }

def calculate_moments_and_props(data, mask, label, wcs=None, obj_slice=None):
    """
    Calculate properties for a specific label in the mask.
    Wrapper around calculate_props_from_indices_values.

    When ``obj_slice`` (the label's bounding box, e.g. from
    ``scipy.ndimage.find_objects``) is supplied, only that sub-volume is scanned
    instead of the whole cube.  Pixel indices are offset back to global
    coordinates, so the moments are identical to the full-scan path (moment sums
    are order-independent) but the cost drops from O(cube) to O(bbox) per label.
    """
    if obj_slice is not None:
        sub_mask = mask[obj_slice] == label
        if not np.any(sub_mask):
            return None
        local = np.where(sub_mask)
        values = data[obj_slice][local]
        indices = tuple(local[i] + obj_slice[i].start for i in range(len(obj_slice)))
    else:
        obj_mask = (mask == label)
        if not np.any(obj_mask):
            return None
        indices = np.where(obj_mask)
        values = data[indices]

    return calculate_props_from_indices_values(indices, values, wcs=wcs, ndim=data.ndim, label_id=label)


def build_catalog(data, mask, wcs=None, labels=None, reporter=None):
    """Build a per-label property catalog efficiently.

    Computes each label's bounding box once via ``scipy.ndimage.find_objects``
    and restricts the per-label moment scan to that box, turning the old
    O(n_labels x cube) cost into roughly O(cube).  Results are identical to
    calling :func:`calculate_moments_and_props` per label on the full cube.
    """
    from scipy.ndimage import find_objects

    if labels is None:
        labels = np.unique(mask)
        labels = labels[labels > 0]

    # find_objects returns a list where entry i is the bounding box of label i+1
    # (None when that label is absent).  One pass over the cube for all labels.
    slices = find_objects(mask)
    n_slices = len(slices)

    catalog = []
    total = len(labels)
    for n, l in enumerate(labels):
        l = int(l)
        obj_slice = slices[l - 1] if 0 < l <= n_slices else None
        props = calculate_moments_and_props(data, mask, l, wcs=wcs, obj_slice=obj_slice)
        if props:
            catalog.append(props)
        if reporter is not None and (n % 32 == 0 or n == total - 1):
            reporter.update(None, f"Building catalog {n + 1}/{total}...")
    return catalog
