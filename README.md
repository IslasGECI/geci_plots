<img src="https://www.islas.org.mx/img/logo.svg" align="right" width="256" />

# geci_plots

A collection of plotting utilities and a Typer-based CLI for generating geographic, temporal, and statistical visualizations from CSV/JSON data. Built on matplotlib, pandas, geopandas, and geoambiental.

## CLI Commands (`geci-plot-cli`)

| Command | Description | Options |
| --- | --- | --- |
| `render-radar-signal-geographic-points` | Bubble chart of radar signal geographic points | `--radar-signal-data-path`, `--global-shapefile-data-path`, `--path-rose-wind`, `--result-map-path` |
| `render-population-time-series` | Time series of captures, births, and population size | `--data-path` (JSON), `--output-path` |
| `render-histogram-with-median` | Histogram with a median line for a CSV column | `--data-path`, `--column-name`, `--output-path` |
| `plot-monthly-traps-effort-and-captures-by-zone` | Monthly trap effort and captures by zone | `--effort-captures-data-path`, `--start-date`, `--end-date`, `--output-path` |
| `plot-monthly-cameras-effort-and-captures` | Monthly camera effort and detections | `--cameras-data-path`, `--start-date`, `--end-date`, `--output-path` |
| `plot-kernel-density-and-points` | Kernel density with overlayed geographic points | `--geographic-data-path`, `--global-shapefile-data-path`, `--path-rose-wind`, `--selected-contour`, `--result-map-path`, `--bandwidth` |
| `plot-kernel-density` | Kernel density only | Same as above |
| `plot-geographic-points` | Geographic points plot | `--geographic-data-path`, `--global-shapefile-data-path`, `--path-rose-wind`, `--result-map-path` |
| `plot-geographic-points-by-trip` | Geographic points colored by trip | Same as above |
| `plot-geographic-points-by-vessel` | Geographic points colored by vessel | Same as above |
| `boxplot` | Boxplot from selected columns (comma-separated) | `--data-path`, `--columns-of-interest`, `--boxplot-path` |
| `version` | Display package version | None |

All plots are saved as transparent images by default.
