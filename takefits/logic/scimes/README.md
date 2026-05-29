# Bundled SCIMES

This directory vendors a modified copy of SCIMES 0.3.3 for use inside
takefits.

Origin: PyPI `scimes==0.3.3`. The source distribution declares `License: BSD`
and includes the BSD 3-Clause license text reproduced here as `LICENSE`.

## License

- SCIMES is BSD 3-Clause (see `LICENSE`).
- No scikit-learn source file is vendored in this directory. The vendored code
  imports scikit-learn as a runtime dependency.

## Notes

- The original `old_spectral_embedding.py` compatibility module is not bundled;
  `scimes.py` uses `sklearn.manifold.spectral_embedding` directly.
- Local patches keep SCIMES working in modern Python/scikit-learn/scipy
  environments, add lazy scikit-learn availability checks, and harden small
  dendrogram / affinity-matrix cases used by Takefits.
