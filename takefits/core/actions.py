import inspect
import importlib
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Optional, get_type_hints

import numpy as np

@dataclass
class Action:
    name: str
    description: str
    handler: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible function schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

class ActionRegistry:
    def __init__(self):
        self._actions: Dict[str, Action] = {}

    def register(self, name: str, description: str, handler: Callable, params_schema: Optional[Dict[str, Any]] = None):
        """
        Register a new action.
        
        Args:
            name: Unique action name.
            description: Human readable description.
            handler: Function to execute.
            params_schema: JSON Schema for parameters. If None, valid from introspection (simplified).
        """
        if params_schema is None:
            params_schema = self._introspect_schema(handler)
            
        action = Action(name=name, description=description, handler=handler, parameters=params_schema)
        self._actions[name] = action

    def get_action(self, name: str) -> Optional[Action]:
        return self._actions.get(name)

    def list_actions(self) -> List[Dict[str, Any]]:
        return [action.to_schema() for action in self._actions.values()]
        
    def execute(self, name: str, **kwargs) -> Any:
        action = self.get_action(name)
        if not action:
            raise ValueError(f"Action '{name}' not found.")
        return action.handler(**kwargs)

    def _introspect_schema(self, func: Callable) -> Dict[str, Any]:
        """Generate a simple JSON schema from type hints."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        properties = {}
        required = []
        
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            Any: "object"
        }

        for param_name, param in sig.parameters.items():
            if param_name in {'state', 'result'}: # Skip injected values
                continue
                
            python_type = type_hints.get(param_name, Any)
            json_type = type_map.get(python_type, "string") # Default to string for complex types
            
            param_schema = {"type": json_type}
            if param.default is not inspect.Parameter.empty:
                param_schema["description"] = f"Default: {param.default}"
            else:
                required.append(param_name)
                
            properties[param_name] = param_schema
            
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

# --- Action Implementation & Registration ---

from takefits.core.app_state import AppState


def _usecases():
    return importlib.import_module("takefits.core.usecases")


def _usecase_handler(name: str) -> Callable[..., Any]:
    def handler(**kwargs):
        return getattr(_usecases(), name)(**kwargs)

    handler.__name__ = name
    handler._lazy_usecase_name = name  # type: ignore[attr-defined]
    return handler


def _build_threshold_mask(state: AppState, threshold: float, condition: str) -> np.ndarray:
    if state.data is None:
        raise ValueError("No data loaded")

    data = state.data
    if np.ma.isMaskedArray(data):
        nan_mask = np.ma.getmaskarray(data)
        data = data.filled(np.nan)
    else:
        nan_mask = np.isnan(data)

    mask = np.ones(data.shape, dtype=np.int16)
    if nan_mask is not None and np.any(nan_mask):
        mask[nan_mask] = 0
    if condition == "less_than":
        mask[data < threshold] = 0
    elif condition == "greater_than":
        mask[data > threshold] = 0
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return mask


def _export_mask_from_threshold(
    state: AppState,
    output_path: str,
    threshold: float,
    condition: str = "less_than",
    history_entries: Optional[List[str]] = None,
    mask_as_float: bool = False,
    nan_for_mask: bool = False,
) -> str:
    mask = _build_threshold_mask(state, threshold, condition)
    return _usecases().export_mask_fits(
        state,
        mask,
        output_path,
        threshold=threshold,
        condition=condition,
        history_entries=history_entries,
        mask_as_float=mask_as_float,
        nan_for_mask=nan_for_mask,
    )


def _apply_regrid(state: AppState, params: Dict[str, Any]) -> AppState:
    result = _usecases().compute_regrid(state, params)
    state.data = result.data
    state.header = result.header
    state.wcs = result.wcs
    return state


def _export_pv_from_result(
    state: AppState,
    result: Any,
    output_path: str,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
    x1: Optional[float] = None,
    y1: Optional[float] = None,
    is_swapped: bool = False,
    history_entries: Optional[List[str]] = None,
    position_origin: str = "start",
    path_type: str = "straight",
    path_length_px: Optional[float] = None,
    x_axis_mode: str = "position",
    phi_start_deg: Optional[float] = None,
    phi_end_deg: Optional[float] = None,
    position_unit: str = "deg",
) -> str:
    if x0 is None:
        x0 = getattr(state, "pv_x0", None)
    if y0 is None:
        y0 = getattr(state, "pv_y0", None)
    if x1 is None:
        x1 = getattr(state, "pv_x1", None)
    if y1 is None:
        y1 = getattr(state, "pv_y1", None)
    if any(value is None for value in (x0, y0, x1, y1)):
        if path_length_px is None:
            raise ValueError("PV export requires slice endpoints or a prior compute_pv action.")
        x0 = y0 = x1 = y1 = 0.0
    return _usecases().export_pv_fits(
        state,
        result,
        output_path,
        x0=float(x0),
        y0=float(y0),
        x1=float(x1),
        y1=float(y1),
        is_swapped=is_swapped,
        history_entries=history_entries,
        position_origin=position_origin,
        path_type=path_type,
        path_length_px=path_length_px,
        x_axis_mode=x_axis_mode,
        phi_start_deg=phi_start_deg,
        phi_end_deg=phi_end_deg,
        position_unit=position_unit,
    )


def _apply_baseline_subtraction_with_result(
    state: AppState,
    world_ranges: List[List[float]],
    order: int = 1,
    reference_pixel: Optional[List[float]] = None,
):
    result = _usecases().compute_polynomial_baseline_subtraction(
        state,
        world_ranges=world_ranges,
        order=order,
        reference_pixel=reference_pixel,
    )
    state.data = result.subtracted_data
    spectral_meta = dict(getattr(state, "spectral_metadata", {}) or {})
    spectral_meta["baseline_last_order"] = int(result.order)
    spectral_meta["baseline_last_world_ranges"] = [list(pair) for pair in result.world_ranges]
    spectral_meta["baseline_last_pixel_ranges"] = [list(pair) for pair in result.pixel_ranges]
    state.spectral_metadata = spectral_meta
    return result

def _wrap_load_fits(filepath: str, hdu: int = 0) -> str:
    # This action updates the GLOBAL state (simulated here for now)
    # In a real agent loop, the agent would hold the state.
    # For now, we assume we return a status message.
    # We might need a context object passed to actions.
    pass

def register_default_actions(registry: ActionRegistry):
    # We need to decide how to handle AppState.
    # For now, let's manually register wrappers that assume 'state' is provided via kwargs 
    # OR we bind them to a specific state instance if we create a session.
    
    # 1. Data Loading
    registry.register(
        name="load_fits",
        description="Load a FITS file.",
        handler=_usecase_handler("load_fits_data"),
        params_schema={
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Absolute path to the FITS file."},
                "hdu": {"type": "integer", "description": "HDU index (default 0)."},
                "compute_wcs": {"type": "boolean", "description": "Whether to compute WCS (default True)."}
            },
            "required": ["filepath"]
        }
    )

    # 2. Slicing
    registry.register(
        name="set_slice",
        description="Set the current Z/S slice indices.",
        handler=_usecase_handler("set_slice"),
        params_schema={
            "type": "object",
            "properties": {
                "z": {"type": "integer", "description": "Z-axis index."},
                "s": {"type": "integer", "description": "S-axis (4th dim) index."}
            },
            "required": []
        }
    )
    
    # 3. Cursor
    registry.register(
        name="set_cursor",
        description="Set the current cursor position (pixel coordinates).",
        handler=_usecase_handler("set_cursor"),
        params_schema={
            "type": "object",
            "properties": {
                "xpix": {"type": "integer", "description": "X coordinate."},
                "ypix": {"type": "integer", "description": "Y coordinate."},
                "zpix": {"type": "integer", "description": "Z coordinate."},
                "spix": {"type": "integer", "description": "S coordinate."}
            },
            "required": []
        }
    )

    # 4. Integrate / Moment
    registry.register(
        name="compute_moment",
        description="Calculate a moment map.",
        handler=_usecase_handler("compute_moment"),
        params_schema={
            "type": "object",
            "properties": {
                "moment_type": {"type": "string", "enum": ["moment0", "moment1", "moment2", "average", "peak", "rms"]},
                "axis": {"type": "integer", "description": "Axis to integrate along (0=z, 1=y, 2=x)."},
                "pixel_range": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2, "maxItems": 2,
                    "description": "(min_pix, max_pix) range along the axis."
                },
                "world_range": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2, "maxItems": 2,
                    "description": "(min_world, max_world) range in world coordinates (e.g., km/s for velocity)."
                },
                "clip_threshold": {"type": "number", "description": "Clip values below this threshold."}
            },
            "required": ["moment_type"]
        }
    )

    # 5. PV Diagram
    registry.register(
        name="compute_pv",
        description="Compute a PV diagram.",
        handler=_usecase_handler("compute_pv"),
        params_schema={
            "type": "object",
            "properties": {
                "x0": {"type": "number"}, "y0": {"type": "number"},
                "x1": {"type": "number"}, "y1": {"type": "number"},
                "width": {"type": "number"},
                "num_samples": {"type": "integer"},
                "sample_spacing_pix": {"type": "number"},
                "weight_mode": {"type": "integer"},
                "path_type": {"type": "string", "enum": ["straight", "polyline", "ellipse", "ellipse_arc"]},
                "center": {"type": "array", "items": {"type": "number"}, "minItems": 2},
                "semi_major_px": {"type": "number"},
                "semi_minor_px": {"type": "number"},
                "pa_rad": {"type": "number"},
                "start_phi_rad": {"type": "number"},
                "end_phi_rad": {"type": "number"},
                "vertices": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2}},
                "vertices_world": {"type": "array", "items": {"type": "array", "minItems": 2}},
                "spline_type": {"type": "string", "enum": ["none", "catmull_rom", "bspline"], "description": "Polyline curve: 'catmull_rom' curves through the nodes, 'bspline' approximates (smooths) them, 'none' keeps straight segments."},
                "smoothness": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Spline roundness 0..1 for a curve spline_type (1=full curve, 0=straight)."},
                "sample_axis": {"type": "string", "enum": ["position", "phi"]}
            },
            "required": []
        }
    )

    # 6. Export
    registry.register(
        name="export_pv_fits",
        description="Export PV data to FITS.",
        handler=_export_pv_from_result,
        params_schema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "x0": {"type": "number"}, "y0": {"type": "number"},
                "x1": {"type": "number"}, "y1": {"type": "number"},
                "is_swapped": {"type": "boolean"},
                "history_entries": {"type": "array", "items": {"type": "string"}},
                "position_origin": {"type": "string", "enum": ["start", "center"]},
                "path_type": {"type": "string", "enum": ["straight", "polyline", "ellipse", "ellipse_arc"]},
                "path_length_px": {"type": "number"},
                "x_axis_mode": {"type": "string", "enum": ["position", "phi"]},
                "phi_start_deg": {"type": "number"},
                "phi_end_deg": {"type": "number"},
                "position_unit": {"type": "string", "enum": ["pixel", "pix", "arcsec", "arcmin", "deg"]},
            },
            "required": ["output_path"]
        }
    )

    registry.register(
        name="export_spectrum",
        description="Export spectrum to text file.",
        handler=_usecase_handler("export_spectrum"),
        params_schema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "xlabel": {"type": "string"},
                "ylabel": {"type": "string"},
                "metadata": {"type": "object"}
            },
            "required": ["output_path"]
        }
    )
    
    registry.register(
        name="export_figure",
        description="Export a matplotlib figure to file.",
        handler=_usecase_handler("export_figure"),
         params_schema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "dpi": {"type": "integer"},
                "transparent": {"type": "boolean"}
            },
            "required": ["output_path"]
        }
    )

    registry.register(
        name="export_data_fits",
        description="Export the current data cube to FITS.",
        handler=_usecase_handler("export_data_fits"),
        params_schema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "history_entries": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["output_path"]
        }
    )

    registry.register(
        name="export_mask_fits",
        description="Export a threshold mask (1=unmasked, 0=masked) to FITS.",
        handler=_export_mask_from_threshold,
        params_schema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "threshold": {"type": "number"},
                "condition": {"type": "string", "enum": ["less_than", "greater_than"]},
                "history_entries": {"type": "array", "items": {"type": "string"}},
                "mask_as_float": {"type": "boolean", "description": "Save mask as float32 instead of int16."},
                "nan_for_mask": {"type": "boolean", "description": "If saving as float, convert masked pixels (0) to NaN."}
            },
            "required": ["output_path", "threshold"]
        }
    )

    registry.register(
        name="get_spectrum",
        description="Get spectrum at a specific pixel coordinate.",
        handler=_usecase_handler("get_spectrum"),
        params_schema={
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X pixel coordinate."},
                "y": {"type": "integer", "description": "Y pixel coordinate."}
            },
            "required": []
        }
    )

    # --- Phase 9a: Tier 1 Usecases ---

    # Color Settings
    registry.register(
        name="set_color_settings",
        description="Update color/display settings for a view plane.",
        handler=_usecase_handler("set_color_settings"),
        params_schema={
            "type": "object",
            "properties": {
                "plane": {"type": "string", "enum": ["xy", "xz", "zy"], "description": "Which plane to update."},
                "vmin": {"type": "number", "description": "Minimum value for color normalization."},
                "vmax": {"type": "number", "description": "Maximum value for color normalization."},
                "cmap": {"type": "string", "description": "Colormap name (e.g., 'viridis')."},
                "log_scale": {"type": "boolean", "description": "Whether to use log scaling."},
                "gamma": {"type": "number", "description": "Gamma correction factor."},
                "invert_cmap": {"type": "boolean", "description": "Whether to invert the colormap."}
            },
            "required": []
        }
    )

    # View Range
    registry.register(
        name="set_view_range",
        description="Set view range for a plane (pixel coordinates).",
        handler=_usecase_handler("set_view_range"),
        params_schema={
            "type": "object",
            "properties": {
                "plane": {"type": "string", "enum": ["xy", "xz", "zy"]},
                "xlim": {"type": "array", "items": {"type": "number"}, "description": "X-axis limits [min, max]."},
                "ylim": {"type": "array", "items": {"type": "number"}, "description": "Y-axis limits [min, max]."}
            },
            "required": []
        }
    )

    # Coordinate grid overlay (TF-404 / TF-407)
    registry.register(
        name="set_coordinate_grid",
        description="Configure the WCS coordinate grid overlay (display only; "
                    "the XY grid can follow a non-native display frame).",
        handler=_usecase_handler("set_coordinate_grid"),
        params_schema={
            "type": "object",
            "properties": {
                "visible": {"type": "boolean", "description": "Whether the coordinate grid is drawn."},
                "frame": {
                    "type": "string",
                    "enum": ["native", "icrs", "fk5", "fk4", "galactic"],
                    "description": "Display frame followed by the XY grid."
                },
                "keep_native": {
                    "type": "boolean",
                    "description": "Keep the native XY grid beneath a non-native overlay."
                }
            },
            "required": ["visible"]
        }
    )

    # Smoothing
    registry.register(
        name="apply_smoothing",
        description="Apply smoothing to the data.",
        handler=_usecase_handler("apply_smoothing"),
        params_schema={
            "type": "object",
            "properties": {
                "kernel_type": {"type": "string", "enum": ["gaussian", "boxcar", "hanning"]},
                "smoothness_x": {"type": "number", "description": "Kernel width in pixels (X)."},
                "smoothness_y": {"type": "number", "description": "Kernel width in pixels (Y)."},
                "smoothness_z": {"type": "number", "description": "Kernel width in pixels (Z)."}
            },
            "required": []
        }
    )

    registry.register(
        name="apply_smoothing_to_resolution",
        description="Apply Gaussian smoothing to target beam resolution.",
        handler=_usecase_handler("apply_smoothing_to_resolution"),
        params_schema={
            "type": "object",
            "properties": {
                "target_bmaj": {"type": "number", "description": "Target BMAJ in arcsec."},
                "target_bmin": {"type": "number", "description": "Target BMIN in arcsec."},
                "target_bpa": {"type": "number", "description": "Target BPA in degrees."},
                "current_bmaj": {"type": "number", "description": "Current BMAJ in arcsec (optional override)."},
                "current_bmin": {"type": "number", "description": "Current BMIN in arcsec (optional override)."},
                "current_bpa": {"type": "number", "description": "Current BPA in degrees (optional override)."},
            },
            "required": ["target_bmaj", "target_bmin"],
        },
    )

    # Scaling
    registry.register(
        name="apply_scaling",
        description="Multiply the current cube by a scalar factor.",
        handler=_usecase_handler("apply_scaling"),
        params_schema={
            "type": "object",
            "properties": {
                "scale_factor": {"type": "number", "description": "Scalar multiplier."}
            },
            "required": ["scale_factor"]
        }
    )

    registry.register(
        name="apply_baseline_subtraction",
        description="Apply polynomial baseline subtraction using world-coordinate line-free ranges.",
        handler=_apply_baseline_subtraction_with_result,
        params_schema={
            "type": "object",
            "properties": {
                "world_ranges": {
                    "type": "array",
                    "description": "List of (min_world, max_world) baseline-fit ranges.",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "order": {
                    "type": "integer",
                    "description": "Polynomial order.",
                    "minimum": 0,
                    "maximum": 9,
                },
                "reference_pixel": {
                    "type": "array",
                    "description": "Optional full WCS reference pixel tuple used for world->pixel conversion.",
                    "items": {"type": "number"},
                },
            },
            "required": ["world_ranges"],
        },
    )

    registry.register(
        name="export_baseline_model_fits",
        description="Export the most recent baseline model to FITS.",
        handler=_usecase_handler("export_baseline_model_fits"),
        params_schema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "history_entries": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["output_path"],
        },
    )

    # Masking
    registry.register(
        name="apply_mask_threshold",
        description="Apply threshold mask to data.",
        handler=_usecase_handler("apply_mask_threshold"),
        params_schema={
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "description": "Threshold value."},
                "condition": {"type": "string", "enum": ["less_than", "greater_than"]}
            },
            "required": ["threshold"]
        }
    )

    registry.register(
        name="apply_mask_external",
        description="Apply external FITS mask to data.",
        handler=_usecase_handler("apply_mask_external"),
        params_schema={
            "type": "object",
            "properties": {
                "mask_path": {"type": "string", "description": "Path to mask FITS file."},
                "mask_value": {"type": "number", "description": "Value indicating masked pixels (default 0)."}
            },
            "required": ["mask_path"]
        }
    )

    registry.register(
        name="apply_mask_moment_recipe",
        description="Apply automatic moment-analysis mask recipe to data.",
        handler=_usecase_handler("apply_mask_moment_recipe"),
        params_schema={
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "enum": ["smoothed_hysteresis", "moment_masking"],
                    "description": "Mask generation recipe.",
                },
                "polarity": {
                    "type": "string",
                    "enum": ["emission", "absorption"],
                    "description": "Line polarity to detect.",
                },
                "preset": {
                    "type": "string",
                    "enum": ["faint", "normal", "strict"],
                    "description": "Preset parameter bundle.",
                },
                "smooth_xy_pix": {"type": "number", "description": "Spatial smoothing sigma [pixel]."},
                "smooth_v_chan": {"type": "number", "description": "Spectral smoothing sigma [channel]."},
                "seed_sigma": {"type": "number", "description": "Seed threshold in sigma (A algorithm)."},
                "grow_sigma": {"type": "number", "description": "Grow threshold in sigma (A algorithm)."},
                "clip_sigma": {"type": "number", "description": "Clip threshold in sigma (B algorithm)."},
                "expand_xy_pix": {"type": "integer", "description": "Spatial expansion radius [pixel] (B algorithm)."},
                "expand_v_chan": {"type": "integer", "description": "Spectral expansion radius [channel] (B algorithm)."},
                "min_channels": {"type": "integer", "description": "Minimum spectral extent for components."},
                "min_voxels": {"type": "integer", "description": "Minimum component voxel count."},
                "connectivity": {
                    "type": "integer",
                    "enum": [6, 18, 26],
                    "description": "3D connectivity for component filtering.",
                },
                "noise_method": {
                    "type": "string",
                    "enum": ["diff_mad", "mad", "std"],
                    "description": "Noise estimator used for sigma thresholds.",
                },
            },
            "required": []
        }
    )

    # Clump Finding
    registry.register(
        name="run_clumpfind",
        description="Run ClumpFind algorithm.",
        handler=_usecase_handler("run_clumpfind"),
        params_schema={
            "type": "object",
            "properties": {
                "rms": {"type": "number", "description": "RMS noise level."},
                "min_threshold_sigma": {"type": "number", "description": "Min threshold in sigma."},
                "step_sigma": {"type": "number", "description": "Contour step size in sigma."},
                "min_pixels": {"type": "integer", "description": "Min pixels per clump."}
            },
            "required": ["rms"]
        }
    )

    registry.register(
        name="run_fellwalker",
        description="Run FellWalker (watershed) algorithm.",
        handler=_usecase_handler("run_fellwalker"),
        params_schema={
            "type": "object",
            "properties": {
                "rms": {"type": "number", "description": "RMS noise level."},
                "min_threshold_sigma": {"type": "number", "description": "Min threshold in sigma."},
                "min_dip_sigma": {"type": "number", "description": "Min dip (prominence) in sigma."},
                "min_pixels": {"type": "integer", "description": "Min pixels per clump."}
            },
            "required": ["rms"]
        }
    )

    registry.register(
        name="run_dendrogram",
        description="Run Dendrogram algorithm.",
        handler=_usecase_handler("run_dendrogram"),
        params_schema={
            "type": "object",
            "properties": {
                "rms": {"type": "number", "description": "RMS noise level."},
                "min_value_sigma": {"type": "number", "description": "Min value in sigma."},
                "min_delta_sigma": {"type": "number", "description": "Min delta in sigma."},
                "min_npix": {"type": "integer", "description": "Min pixels per structure."},
                "output_mode": {"type": "string", "enum": ["leaves", "roots", "all"]},
                "use_scimes": {"type": "boolean", "description": "Whether to use SCIMES."}
            },
            "required": ["rms"]
        }
    )

    registry.register(
        name="export_clump_mask",
        description="Export clump mask to FITS.",
        handler=_usecase_handler("export_clump_mask"),
        params_schema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string", "description": "Output FITS file path."},
                "history_entries": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["output_path"]
        }
    )

    registry.register(
        name="export_clump_catalog",
        description="Export clump catalog to CSV or FITS.",
        handler=_usecase_handler("export_clump_catalog"),
        params_schema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string", "description": "Output file path."},
                "format": {"type": "string", "enum": ["csv", "fits"]}
            },
            "required": ["output_path"]
        }
    )

    # --- Phase 9b: Tier 2 Usecases ---

    # Cutout
    registry.register(
        name="compute_cutout",
        description="Extract a cutout from the data cube.",
        handler=_usecase_handler("compute_cutout"),
        params_schema={
            "type": "object",
            "properties": {
                "pixel_bounds": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "description": "List of (start, stop) for each WCS axis [X, Y, Z, ...]."
                },
                "world_bounds": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "description": "List of (min, max) in world coordinates for each axis."
                }
            },
            "required": []
        }
    )

    registry.register(
        name="export_cutout_fits",
        description="Export cutout result to FITS file.",
        handler=_usecase_handler("export_cutout_fits"),
        params_schema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string", "description": "Output FITS file path."},
                "source_filename": {"type": "string", "description": "Original source filename."},
                "history_entries": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["output_path"]
        }
    )

    # Arithmetic
    registry.register(
        name="compute_arithmetic",
        description="Perform arithmetic operation on data arrays.",
        handler=_usecase_handler("apply_arithmetic"),
        params_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "expression"],
                    "description": "Operation type."
                },
                "data_b_path": {"type": "string", "description": "Optional FITS path used as variable B."},
                "expression": {"type": "string", "description": "NumPy expression (for 'expression' mode)."},
                "scalar": {"type": "number", "description": "Scalar value for simple operations."}
            },
            "required": ["operation"]
        }
    )

    # Channel Map
    registry.register(
        name="compute_channel_map",
        description="Generate channel map images from a data cube.",
        handler=_usecase_handler("compute_channel_map"),
        params_schema={
            "type": "object",
            "properties": {
                "start_channel": {"type": "integer", "description": "Starting channel (0-indexed)."},
                "end_channel": {"type": "integer", "description": "Ending channel (exclusive)."},
                "interval": {"type": "integer", "description": "Channels per panel."},
                "mode": {"type": "string", "enum": ["slice", "average", "integrate"]},
                "axis": {"type": "integer", "description": "Axis for channels (0=z)."},
                "start_world": {"type": "number", "description": "Starting position in world coords."},
                "end_world": {"type": "number", "description": "Ending position in world coords."},
                "interval_world": {"type": "number", "description": "Interval in world coords (e.g., km/s)."}
            },
            "required": []
        }
    )

    # --- Phase 9c: Tier 3 Usecases ---

    registry.register(
        name="compute_regrid",
        description="Regrid the current data cube using RegridPanel parameters.",
        handler=_apply_regrid,
        params_schema={
            "type": "object",
            "properties": {
                "params": {"type": "object", "description": "RegridPanel parameter dict."}
            },
            "required": ["params"]
        }
    )

    # Unit Conversion
    registry.register(
        name="convert_intensity_unit",
        description="Convert intensity units (Jy/beam, K, Jy/pixel).",
        handler=_usecase_handler("apply_unit_conversion"),
        params_schema={
            "type": "object",
            "properties": {
                "from_unit": {
                    "type": "string",
                    "enum": ["jy/beam", "k", "jy/pix"],
                    "description": "Source unit."
                },
                "to_unit": {
                    "type": "string",
                    "enum": ["jy/beam", "k", "jy/pix"],
                    "description": "Target unit."
                },
                "method": {
                    "type": "string",
                    "enum": ["rayleigh-jeans", "planck"],
                    "description": "Conversion method."
                }
            },
            "required": ["from_unit", "to_unit"]
        }
    )

    # --- Phase 9d: Visualization Export ---

    registry.register(
        name="export_moment_image",
        description="Compute and export moment map as PNG image.",
        handler=_usecase_handler("export_moment_image"),
        params_schema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "moment_type": {"type": "string", "enum": ["moment0", "moment1", "moment2", "average", "peak"]},
                "axis": {"type": "integer"},
                "cmap": {"type": "string"},
                "pixel_range": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2, "maxItems": 2
                },
                "title": {"type": "string"},
                "grid": {"type": "boolean", "description": "Draw the WCS coordinate grid overlay (TF-404)."},
                "grid_frame": {
                    "type": "string",
                    "enum": ["native", "icrs", "fk5", "fk4", "galactic"],
                    "description": "Display frame followed by the XY grid (TF-407)."
                },
                "grid_keep_native": {
                    "type": "boolean",
                    "description": "Keep the native grid beneath a non-native XY overlay."
                }
            },
            "required": ["output_path"]
        }
    )

    registry.register(
        name="export_moment_fits",
        description="Compute and export moment map as FITS file.",
        handler=_usecase_handler("export_moment_map_fits"),
        params_schema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "moment_type": {"type": "string", "enum": ["moment0", "moment1", "moment2", "average", "peak"]},
                "axis": {"type": "integer"},
                "pixel_range": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2, "maxItems": 2
                },
                "world_range": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2, "maxItems": 2
                },
                "history_entries": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "display_fits_axes": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2, "maxItems": 2
                }
            },
            "required": ["output_path"]
        }
    )

    registry.register(
        name="export_channel_map_image",
        description="Compute and export channel map grid as PNG image.",
        handler=_usecase_handler("export_channel_map_image"),
        params_schema={
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "start_channel": {"type": "integer"},
                "end_channel": {"type": "integer"},
                "interval": {"type": "number"},
                "mode": {"type": "string", "enum": ["slice", "average", "integrate"]},
                "ncols": {"type": "integer"},
                "cmap": {"type": "string"}
            },
            "required": ["output_path"]
        }
    )

    # --- Annotations (Regions / Markers) ---

    registry.register(
        name="set_regions",
        description="Replace the current region list (small snapshot).",
        handler=_usecase_handler("set_regions"),
        params_schema={
            "type": "object",
            "properties": {
                "regions": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of region specs (dicts).",
                }
            },
            "required": ["regions"],
        },
    )

    registry.register(
        name="clear_regions",
        description="Remove all regions from state.",
        handler=_usecase_handler("clear_regions"),
        params_schema={"type": "object", "properties": {}, "required": []},
    )

    registry.register(
        name="add_region",
        description="Add a single region to state.",
        handler=_usecase_handler("add_region"),
        params_schema={
            "type": "object",
            "properties": {
                "region": {"type": "object", "description": "Region spec (dict)."}
            },
            "required": ["region"],
        },
    )

    registry.register(
        name="update_region",
        description="Update an existing region in state.",
        handler=_usecase_handler("update_region"),
        params_schema={
            "type": "object",
            "properties": {
                "region_id": {"type": "string", "description": "Region id to update."},
                "updates": {"type": "object", "description": "Partial region spec fields."},
            },
            "required": ["region_id", "updates"],
        },
    )

    registry.register(
        name="delete_region",
        description="Delete an existing region from state.",
        handler=_usecase_handler("delete_region"),
        params_schema={
            "type": "object",
            "properties": {
                "region_id": {"type": "string", "description": "Region id to delete."}
            },
            "required": ["region_id"],
        },
    )

    registry.register(
        name="set_markers",
        description="Replace the current marker list (small snapshot).",
        handler=_usecase_handler("set_markers"),
        params_schema={
            "type": "object",
            "properties": {
                "markers": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of marker specs (dicts).",
                }
            },
            "required": ["markers"],
        },
    )

    registry.register(
        name="clear_markers",
        description="Remove all markers from state.",
        handler=_usecase_handler("clear_markers"),
        params_schema={"type": "object", "properties": {}, "required": []},
    )

    registry.register(
        name="add_marker",
        description="Add a single marker to state (upsert by id).",
        handler=_usecase_handler("add_marker"),
        params_schema={
            "type": "object",
            "properties": {
                "marker": {"type": "object", "description": "Marker spec (dict)."}
            },
            "required": ["marker"],
        },
    )

    registry.register(
        name="update_marker",
        description="Update an existing marker in state.",
        handler=_usecase_handler("update_marker"),
        params_schema={
            "type": "object",
            "properties": {
                "marker_id": {"type": "string", "description": "Marker id to update."},
                "updates": {"type": "object", "description": "Partial marker spec fields."},
            },
            "required": ["marker_id", "updates"],
        },
    )

    registry.register(
        name="delete_marker",
        description="Delete an existing marker from state.",
        handler=_usecase_handler("delete_marker"),
        params_schema={
            "type": "object",
            "properties": {
                "marker_id": {"type": "string", "description": "Marker id to delete."}
            },
            "required": ["marker_id"],
        },
    )
