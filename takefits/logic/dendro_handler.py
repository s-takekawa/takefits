from astrodendro import Dendrogram, pp_catalog, ppv_catalog
from astropy import units as u
from astropy.table import Column, Table
import numpy as np
import warnings

_ASTRODENDRO_BOOL_PATCHED = False


def _patch_astrodendro_quantity_truthiness():
    """Patch astrodendro properties to avoid Quantity truthiness checks."""
    global _ASTRODENDRO_BOOL_PATCHED
    if _ASTRODENDRO_BOOL_PATCHED:
        return
    try:
        import astrodendro.analysis as analysis
    except Exception:
        return

    def _safe_scale(value):
        return value if value is not None else u.pixel

    def _major_sigma(self):
        dx = _safe_scale(self.spatial_scale)
        a, _b = self._sky_paxes()
        return dx * np.sqrt(self.stat.mom2_along(tuple(a)))

    def _minor_sigma(self):
        dx = _safe_scale(self.spatial_scale)
        _a, b = self._sky_paxes()
        return dx * np.sqrt(self.stat.mom2_along(tuple(b)))

    def _v_rms(self):
        dv = self.velocity_scale if self.velocity_scale is not None else u.pixel
        ax = [0, 0, 0]
        ax[self.vaxis] = 1
        return dv * np.sqrt(self.stat.mom2_along(tuple(ax)))

    def _area_exact(self):
        dx = _safe_scale(self.spatial_scale)
        indices = zip(*tuple(self.stat.indices[i] for i in range(3) if i != self.vaxis))
        return len(set(indices)) * dx ** 2

    analysis.SpatialBase.major_sigma = property(_major_sigma)
    analysis.SpatialBase.minor_sigma = property(_minor_sigma)
    analysis.PPVStatistic.v_rms = property(_v_rms)
    analysis.PPVStatistic.area_exact = property(_area_exact)
    _ASTRODENDRO_BOOL_PATCHED = True

class DendroHandler:
    @staticmethod
    def scimes_unavailable_reason():
        """Return why SCIMES cannot run, or an empty string when available."""
        try:
            from takefits.logic.scimes.scimes import _require_sklearn
            _require_sklearn()
            return ""
        except ImportError as exc:
            return str(exc)

    @staticmethod
    def is_scimes_available():
        """Check if SCIMES module is available."""
        return DendroHandler.scimes_unavailable_reason() == ""

    def __init__(self, data, wcs=None, header=None):
        self.data = data
        self.wcs = wcs
        self.header = header
        self.d = None
        self.leaves = []
        self.clusters = []
        self.catalog_cache = None

    @staticmethod
    def _classify_axis_type(ctype: str) -> str:
        """Return a simple classification for a FITS CTYPE value."""
        if not ctype:
            return 'unknown'
        upper = str(ctype).upper()
        if 'FREQ' in upper:
            return 'frequency'
        if any(tag in upper for tag in ('VRAD', 'VELO', 'VOPT')):
            return 'velocity'
        return 'unknown'

    def _identify_spectral_axis(self) -> int | None:
        """Identify the first spectral-like FITS axis (1-based)."""
        if self.header is None:
            return None
        try:
            naxis = int(self.header.get('NAXIS', 0))
        except (TypeError, ValueError):
            return None
        for axis in range(1, naxis + 1):
            ctype = self.header.get(f'CTYPE{axis}', '')
            if self._classify_axis_type(ctype) != 'unknown':
                return axis
        return None

    def _build_catalog_metadata(self):
        """Build astrodendro metadata for catalog generation."""
        metadata = {'data_unit': u.Jy / u.beam, 'wavelength': 1.0 * u.mm}

        if self.wcs:
            from astropy.wcs.utils import proj_plane_pixel_scales
            scales = proj_plane_pixel_scales(self.wcs)
            if len(scales) >= 2:
                avg_scale = np.sqrt(scales[0] * scales[1])
                try:
                    is_quantity = isinstance(1 * avg_scale, u.Quantity)
                except Exception:
                    is_quantity = False
                if is_quantity:
                    metadata['spatial_scale'] = avg_scale
                else:
                    metadata['spatial_scale'] = float(avg_scale) * u.deg
            elif self.header and 'CDELT2' in self.header:
                metadata['spatial_scale'] = abs(float(self.header['CDELT2'])) * u.deg
            else:
                metadata['spatial_scale'] = None

        if self.header:
            bmaj = self.header.get('BMAJ')
            bmin = self.header.get('BMIN')
            if bmaj:
                metadata['beam_major'] = float(bmaj) * u.deg
            else:
                metadata['beam_major'] = 0.01 * u.deg

            if bmin:
                metadata['beam_minor'] = float(bmin) * u.deg
            else:
                metadata['beam_minor'] = metadata['beam_major']
        else:
            metadata['beam_major'] = 0.01 * u.deg
            metadata['beam_minor'] = 0.01 * u.deg

        if self.data.ndim == 3:
            metadata['velocity_scale'] = 1.0 * u.km / u.s
            spec_axis = self._identify_spectral_axis()
            if spec_axis is not None:
                vaxis = self.data.ndim - int(spec_axis)
                if 0 <= vaxis < self.data.ndim:
                    metadata['vaxis'] = int(vaxis)

        return metadata

    def _build_native_catalog_table(self):
        """Safely build an astrodendro catalog table."""
        if self.d is None:
            return None

        try:
            if len(self.d) == 0:
                return Table()
        except Exception:
            pass

        metadata = self._build_catalog_metadata()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                if self.data.ndim == 3:
                    cat = ppv_catalog(self.d, metadata)
                else:
                    cat = pp_catalog(self.d, metadata)
            except AttributeError as exc:
                if "'NoneType' object has no attribute 'sort'" not in str(exc):
                    raise
                cat = None

        return cat if cat is not None else Table()

    def run_dendrogram(self, min_value, min_delta, min_npix):
        """
        Run astrodendro.
        """
        try:
            self.d = Dendrogram.compute(self.data, min_value=min_value, min_delta=min_delta, min_npix=min_npix, wcs=self.wcs, verbose=True)
            self.leaves = self.d.leaves
            self.catalog_cache = None
            return len(self.leaves)
        except Exception as e:
            print(f"Dendrogram Error: {e}")
            return 0

    def run_scimes(self, save_isol_leaves=True, criteria=None, criteria_weights=None, rms=np.nan, s2nlim=3, locscaling=False, user_k=0):
        """
        Run SCIMES on top of the dendrogram.
        save_isol_leaves: If True, include isolated leaves in the results.
        criteria: List of criteria ['volume', 'luminosity']. If None, defaults based on dim.
        criteria_weights: Optional list/dict of weights for criteria aggregation.
        rms: Noise level (same units as data) for SCIMES scaling parameter estimation.
        s2nlim: Signal-to-noise limit used when estimating scaling parameters.
        locscaling: If True, use local scaling for affinity matrix smoothing.
        user_k: User-specified number of clusters (approximate target). 0 for auto.
        """
        if self.d is None:
            return False, "Run Dendrogram first."

        try:
            structure_count = len(self.d)
            leaf_count = len(self.leaves)
            if structure_count == 0 or leaf_count == 0:
                self.clusters = []
                return True, "No dendrogram structures found; SCIMES was skipped."

            # SCIMES assumes a non-trivial affinity matrix. Mirror its intended
            # small-N fallback locally instead of letting the vendored code fail.
            if leaf_count <= 2:
                self.clusters = list(self.leaves)
                return True, f"Only {leaf_count} dendrogram leaves found; each leaf was kept as its own cluster."

            unavailable_reason = self.scimes_unavailable_reason()
            if unavailable_reason:
                return False, unavailable_reason

            _patch_astrodendro_quantity_truthiness()
            # Import SCIMES from the bundled location
            from takefits.logic.scimes import SpectralCloudstering

            # Default criteria if not provided
            if criteria is None or len(criteria) == 0:
                if self.data.ndim == 3:
                    criteria = ['volume', 'luminosity']
                else:
                    criteria = ['area_exact', 'luminosity']

            # Generate catalog
            # Use cached catalog if available to speed up re-runs with different SCIMES params
            if self.catalog_cache is not None:
                cat = self.catalog_cache
            else:
                cat = self._build_native_catalog_table()
                self.catalog_cache = cat

            if cat is None or len(cat) == 0:
                self.clusters = []
                return True, "SCIMES was skipped because the dendrogram catalog is empty."

            # Strip units from catalog for SCIMES compatibility
            # SCIMES 0.3.2+ / astrodendro interaction issues can cause scaling parameters to be ~1.0
            # if Quantity objects are passed. We ensure pure floats are used.
            # We work on a copy to avoid modifying the cached catalog if it's reused elsewhere
            scimes_cat = cat.copy()
            for col in scimes_cat.colnames:
                if hasattr(scimes_cat[col], 'unit') and scimes_cat[col].unit is not None:
                     scimes_cat[col] = scimes_cat[col].value
            
            # SCIMES interacts with the dendrogram object directly to assign 'clusters'
            # init(dendrogram, catalog, header)

            # Handle header: Fscimes passes the FITS header, so we should too.
            # Even if catalog is pixel-based, SCIMES might look at BMAJ/BMIN in the header if available.
            header_to_use = self.header
            if header_to_use is None:
                from astropy.io import fits
                header_to_use = fits.Header()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
                    dclust = SpectralCloudstering(
                        self.d,
                        scimes_cat, # Pass unit-stripped catalog
                        header_to_use,
                        criteria=criteria,
                        criteria_weights=criteria_weights,
                        save_all_leaves=save_isol_leaves,  # Use save_all_leaves for completeness
                        rms=rms,
                        s2nlim=s2nlim,
                        locscaling=locscaling,
                        user_k=user_k, # Pass user_k to SCIMES
                    )
            dclust.show = False # Disable matplotlib popup
            # dclust.run_clustering() # Run (Called automatically in __init__)

            # The result is stored in dclust.clusters (list of indices)
            # Convert indices to structure objects
            self.clusters = [self.d[idx] for idx in dclust.clusters]

            # We can also get a mask of clusters
            # mask_cube = dclust.get_clusters_mask() # This might be new scimes version dependent
            return True, f"Found {len(self.clusters)} clusters (Isolated included: {save_isol_leaves})."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)

    def get_mask(self, mode='leaves'):
        """
        Return a mask of the structures.
        mode: 'leaves' (default dendro leaves) or 'clusters' (scimes clusters)
        """
        if self.d is None:
            return np.zeros_like(self.data, dtype=int)

        mask = np.zeros_like(self.data, dtype=int)

        target_structures = []
        if mode == 'leaves':
            target_structures = self.leaves
        elif mode == 'clusters':
            target_structures = self.clusters
        elif mode == 'roots':
            target_structures = self.d.trunk
        elif mode == 'all':
            # Flattened hierarchy: Branches won't be filled, they will be shells around children.
            # We must paint PARENTS (larger) first, then CHILDREN (smaller).
            # Sorting by index is risky. Sorting by area/volume is safer.
            all_structs = list(self.d.all_structures)
            # Sort by number of pixels (descending) so large structures are painted first
            # struct.get_mask().sum() is expensive? struct.indices is fast?
            # struct.npix is available? Yes, struct.get_npix() or similar?
            # Astrodendro structures usually have properties computed.
            # Let's rely on the fact that parents are always larger than children.
            # indices is a method, so we must call it.
            target_structures = sorted(all_structs, key=lambda s: len(s.indices()), reverse=True)
        else:
            target_structures = self.leaves

        # Assign IDs:
        # We want stable IDs if possible?
        # For 'all', using struct.idx is natural.
        # For others, 1..N is standard.
        
        counter = 1
        for struct in target_structures:
            m = struct.get_mask()
            
            if mode == 'all':
                # Use the dendrogram structure ID directly for traceability
                idx = struct.idx + 1 # 1-based
                mask[m] = idx
            else:
                mask[m] = counter
                counter += 1

        return mask

    def get_catalog(self, mode='leaves'):
        """
        Generate a catalog using standardized logic.
        """
        if self.d is None:
            return []

        target_structures = []
        if mode == 'leaves':
            target_structures = self.leaves
        elif mode == 'clusters':
            target_structures = self.clusters
        elif mode == 'roots':
            target_structures = self.d.trunk
        # For 'all', we don't need a linear list for lookup if we use idx directly

        from takefits.logic.cloud_catalog_utils import calculate_moments_and_props

        mask = self.get_mask(mode=mode)
        labels = np.unique(mask)
        labels = labels[labels > 0]

        catalog = []
        for l in labels:
            props = calculate_moments_and_props(self.data, mask, l, self.wcs)

            if props:
                # Identify the structure to add extra metadata (dendro_idx)
                struct = None
                if mode == 'all':
                    # In 'all' mode, label 'l' is (struct.idx + 1)
                    # So struct id is l-1. Dendrogram allows access by integer ID.
                    try:
                        struct = self.d[int(l - 1)]
                    except Exception:
                        pass
                else:
                    # In other modes, label 'l' is 1-based index into target_structures
                    if 0 <= l-1 < len(target_structures):
                        struct = target_structures[l-1]

                if struct:
                     props['dendro_idx'] = struct.idx
                     # Maybe add level or is_leaf info?
                     props['is_leaf'] = struct.is_leaf
                     props['is_branch'] = should_be_branch = not struct.is_leaf
                
                catalog.append(props)

        return catalog

        return catalog

    def get_native_catalog(self):
        """
        Return the native Astrodendro catalog (astropy Table).
        Computed on demand if not cached.
        """
        if self.d is None:
            return None
        
        if self.catalog_cache is not None:
            return self.catalog_cache

        cat = self._build_native_catalog_table()
        if cat is None:
            return None
        if len(cat) == 0:
            self.catalog_cache = cat
            return cat
        
        # Add topological classification
        # We need to map back from catalog index to Structure object.
        # Astrodendro catalog usually has '_idx' column which is the index into self.d
        if '_idx' in cat.colnames and self.d is not None:
             is_leaf_col = []
             is_branch_col = []
             is_trunk_col = []
             type_tag_col = []
             peak_val_col = []
             mean_val_col = []
             npix_col = []
             
             # Prepare columns for unified catalog
             from takefits.logic.cloud_catalog_utils import calculate_props_from_indices_values
             
             # Collect all properties
             rows_data = []
             structure_types = []
             
             for row in cat:
                 idx = row['_idx']
                 struct = self.d[idx]
                 
                 # Topological classification
                 is_leaf = struct.is_leaf
                 has_parent = (struct.parent is not None)
                 
                 if is_leaf and not has_parent:
                     tag = 'isolated'
                 elif is_leaf:
                     tag = 'leaf'
                 elif not has_parent:
                     tag = 'trunk'
                 else:
                     tag = 'branch'
                 structure_types.append(tag)
                 
                 # Calculate physical properties using shared logic
                 indices = struct.indices()
                 values = struct.values()
                 
                 props = calculate_props_from_indices_values(
                     indices, values, wcs=self.wcs, ndim=self.data.ndim, label_id=idx
                 )
                 
                 if props is None:
                     # Handle empty structure (should not happen in dendrogram generally)
                     props = {}
                 
                 rows_data.append(props)
             
             # Convert list of dicts to columns
             if len(rows_data) > 0:
                 keys = rows_data[0].keys()
                 for key in keys:
                     # Skip 'id' if we want to rely on _idx or if it conflicts
                     if key == 'id': continue
                     
                     col_data = [r.get(key, np.nan) for r in rows_data]
                     cat.add_column(Column(col_data, name=key))
             
             cat.add_column(Column(structure_types, name='structure_type'))


        self.catalog_cache = cat
        return cat
