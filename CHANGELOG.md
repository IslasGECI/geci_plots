# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added

### Fixed

### Changed

### Removed

## [0.11.0] - 2026-04-07
### Added
- CLI command `render-histogram-with-median` to write an histogram of a CSV file given a column name.
- CLI command `render-population-time-series` to write a time series of captures, births and population size interval given a posterior summary JSON file.

## [0.10.0] - 2026-03-03
### Added
- CLI command `plot-monthly-traps-effort-and-captures-by-zone` to plot montlhy traps effort and captures by zone.

### Fixed
- All plots generated from CLI commands are transparent.

## [0.9.0] - 2026-02-12
### Added
- CLI command `plot-monthly-cameras-effort-and-captures` to plot montlhy cameras effort and detections.

## [0.8.0] - 2025-07-16
### Added
- Arguments `dpi` and `font_family` to function `geci_plots`.

## [0.7.0] - 2025-03-28

### Added
- Command `plot-geographic-points-by-vessel` for geographic point plot

### Changed
- Command `plot-kernel-density` now have fixed axis limits to show area of interest

## [0.6.0] - 2025-03-27

### Added
- Command `plot-geographic-points-by-trip` for geographic point plot

### Fixed
- Function `format_plot()` now calculates axis limits from the data.


## [0.5.0] - 2025-03-25

### Added
- Cli entrypoint `geci-plot-cli`
  - Command `plot-kernel-density-and-points` for kernel and geographic point plot
  - Command `plot-kernel-density` for kernel plot
  - Command `plot-geographic-points` for geographic point plot

## [0.4.1] - 2023-06-23

### Added

### Fixed

### Changed

### Removed


[unreleased]: https://github.com/IslasGECI/dimorfismo_py/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/IslasGECI/dimorfismo_py/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/IslasGECI/dimorfismo_py/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/IslasGECI/dimorfismo_py/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/IslasGECI/dimorfismo_py/releases/tag/v0.4.1
