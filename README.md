# Takefits

[![PyPI version](https://img.shields.io/pypi/v/takefits)](https://pypi.org/project/takefits/)

Takefits is a GUI-based astronomical FITS viewer and analysis tool developed by Shunya Takekawa. <a href="https://orcid.org/0000-0001-8147-6817"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" alt="ORCID iD" /></a>

## Requirements

- Python 3.11 or later

## Setup

From PyPI:

```bash
pip install takefits
```

For a local checkout:

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install .
```

## Usage

```bash
takefits [path/to/fitsfile]
```

You can also launch it with:

```bash
python -m takefits [path/to/fitsfile]
```

Takefits can also save your current work using workspace files, so you can resume it anytime.
To restore a saved workspace, launch Takefits with the workspace file path.

```bash
takefits /path/to/yourfile.workspace.json
```

## Features

Takefits provides a comprehensive set of tools for radio astronomy data analysis.

### 1. Multi-View Cube Visualization

Visualize 3D FITS data cubes with synchronized XY, XZ, and ZY planes.

![Main Window](https://raw.githubusercontent.com/s-takekawa/takefits/main/docs/images/main_window.png)

### 2. Moment Maps & Channel Maps

Calculate moment maps (Integrated Intensity, Velocity Field, Velocity Dispersion) and create tiled channel maps.

![Channel Map](https://raw.githubusercontent.com/s-takekawa/takefits/main/docs/images/channel_map.png)

### 3. Interactive P-V Diagram

Interactively draw extraction paths — straight lines, polylines, smooth spline curves, and elliptical arcs — to generate Position-Velocity (P-V) diagrams instantly.

![PV Diagram](https://raw.githubusercontent.com/s-takekawa/takefits/main/docs/images/pv_diagram2.png)

### 4. Spectrum Analysis

Extract spectra from a single pixel or calculate the average spectrum within selected regions (Circle, Rectangle, Ellipse, Cube).

![Spectrum](https://raw.githubusercontent.com/s-takekawa/takefits/main/docs/images/spectrum.png)

### 5. Publication-Quality Figures

Generate publication-quality figures directly from the GUI.

- **Contours**: Overlay customizable contours with adjustable levels. Contours can also be loaded from another FITS file with a live preview; the dialog shows data range/rms, lets you generate linear Min/Max/Steps levels or edit the Levels field directly, and aligns the resulting contours through the source WCS so images on different grids can be compared without regridding. Double-click an imported overlay in the contour panel to re-edit its levels.
- **Markers**: Annotate images with symbols, lines, and text.
- **Beam Size**: Visualize the HPBW ellipse.
- **Vector Export**: Save plots in PDF, EPS, or SVG formats.

![Contour Plot](https://raw.githubusercontent.com/s-takekawa/takefits/main/docs/images/contour_example.png)

### Other Tools

- **Regridding**: Resample data to a new grid, different coordinate system, or a FITS template.
- **Smoothing**: Apply Gaussian/Boxcar spatial smoothing and Hanning smoothing along the velocity axis.
- **Masking**: Apply threshold-based or external masks to data.
- **Cutout**: Crop data cubes based on regions or coordinate ranges.
- **Image Arithmetic**: Perform mathematical operations between FITS cubes or with constants using numpy-like expressions.
- **Unit Conversion**: Convert spectral or intensity units for analysis.
- **Baseline Subtraction**: Fit and subtract spectral baselines from data cubes.
- **Clump Finding**: Identify structures using ClumpFind, FellWalker, Dendrogram, and SCIMES (dendrogram clustering) algorithms.

## Research use

If you use this software in scientific publications, please cite the Zenodo DOI:

https://doi.org/10.5281/zenodo.18328843

For a specific release, cite the version DOI listed on Zenodo.

## Algorithm Citations

The tools below implement established analysis algorithms. If you use them for your analysis, please cite the original papers:

### Clump Finding

Cite the paper corresponding to the algorithm you used:

- **ClumpFind** — Williams, J. P., de Geus, E. J., & Blitz, L. (1994). Determining structure in molecular clouds. *The Astrophysical Journal*, 428, 693-712. [doi:10.1086/174279](https://doi.org/10.1086/174279)
- **FellWalker** — Berry, D. S. (2015). FellWalker-A clump identification algorithm. *Astronomy and Computing*, 10, 22-31. [doi:10.1016/j.ascom.2014.11.004](https://doi.org/10.1016/j.ascom.2014.11.004)
- **Dendrogram** — Rosolowsky, E. W., Pineda, J. E., Kauffmann, J., & Goodman, A. A. (2008). Structural Analysis of Molecular Clouds: Dendrograms. *The Astrophysical Journal*, 679(2), 1338-1351. [doi:10.1086/587685](https://doi.org/10.1086/587685)
- **SCIMES** (Dendrogram clustering) — Colombo, D., Rosolowsky, E., Ginsburg, A., Duarte-Cabral, A., & Hughes, A. (2015). Graph-based interpretation of the molecular interstellar medium segmentation. *Monthly Notices of the Royal Astronomical Society*, 454(2), 2067-2091. [doi:10.1093/mnras/stv2063](https://doi.org/10.1093/mnras/stv2063) — a modified copy is bundled under [`takefits/logic/scimes/`](takefits/logic/scimes/) (BSD 3-Clause).

### Masking / Moment Analysis

- Rosolowsky, E., & Leroy, A. (2006). Bias-free Measurement of Giant Molecular Cloud Properties. *Publications of the Astronomical Society of the Pacific*, 118(842), 590-610. [doi:10.1086/502982](https://doi.org/10.1086/502982)
- Dame, T. M. (2011). The Technique of Velocity-Resolved Moment Masking. *arXiv e-prints*, arXiv:1101.1499. [arXiv:1101.1499](https://arxiv.org/abs/1101.1499)

## Contact

shunya_at_kanagawa-u.ac.jp
