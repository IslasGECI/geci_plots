<img src="https://www.islas.org.mx/img/logo.svg" align="right" width="256" />

# geci_plots

One command, one figure: turn monitoring data into images ready for a report.

## What it does

`geci_plots` reads the CSV and JSON files produced by GECI's data pipelines and draws the figure a report, a paper, or a talk needs. Every figure is written as a transparent image, so it sits on any background without a white box.

It draws four families of figures:

- **Where things happened:** bubble charts and maps of observations, coloured by trip or by vessel.
- **When things happened:** monthly effort and capture series for traps and cameras, and population trends.
- **How the data is distributed:** histograms with a median line, boxplots, and kernel density maps.
- **Coverage:** the percentage of each species covering each quadrant of an enclosure.

## Usage (`geci-plot-cli`)

| Command | Description | Options |
| --- | --- | --- |
| `render-radar-signal-geographic-points` | Bubble chart of radar signal geographic points | `--radar-signal-data-path`, `--global-shapefile-data-path`, `--path-rose-wind`, `--result-map-path` |
| `render-population-time-series` | Time series of captures, births, and population size | `--data-path` (JSON), `--output-path` |
| `render-histogram-with-median` | Histogram with a median line for a CSV column | `--data-path`, `--column-name`, `--output-path` |
| `render-species-percentage-coverage-by-quadrant` | Stacked bar chart of species percentage coverage by quadrant for a given enclosure | `--data-path`, `--enclosure`, `--output-path` |
| `plot-monthly-traps-effort-and-captures-by-zone` | Monthly trap effort and captures by zone | `--effort-captures-data-path`, `--start-date`, `--end-date`, `--output-path` |
| `plot-monthly-cameras-effort-and-captures` | Monthly camera effort and detections | `--cameras-data-path`, `--start-date`, `--end-date`, `--output-path` |
| `plot-kernel-density-and-points` | Kernel density with overlayed geographic points | `--geographic-data-path`, `--global-shapefile-data-path`, `--path-rose-wind`, `--selected-contour`, `--result-map-path`, `--bandwidth` |
| `plot-kernel-density` | Kernel density only | Same as above |
| `plot-geographic-points` | Geographic points plot | `--geographic-data-path`, `--global-shapefile-data-path`, `--path-rose-wind`, `--result-map-path` |
| `plot-geographic-points-by-trip` | Geographic points colored by trip | Same as above |
| `plot-geographic-points-by-vessel` | Geographic points colored by vessel | Same as above |
| `boxplot` | Boxplot from selected columns (comma-separated) | `--data-path`, `--columns-of-interest`, `--boxplot-path` |
| `version` | Display package version | None |

Every plotting command saves a transparent image; `version` only prints the package version. Run `geci-plot-cli <command> --help` to see the options of a single command.

## Before you start

- The data file for the figure you want. The Options column says which file each command expects: a CSV exported from our databases, or a JSON summary.
- The folder where the image should be written.
- Docker, or Python 3.9 or newer.

## Run the project

### With Docker (recommended)

From the folder that contains the repository:

```bash
docker compose run --rm islasgeci bash -c "make install && geci-plot-cli version"
```

Replace `version` with the command you need, for example:

```bash
docker compose run --rm islasgeci bash -c "make install && geci-plot-cli boxplot --data-path captures.csv --columns-of-interest weight,length --boxplot-path boxplot.png"
```

### Without Docker

```bash
pip install geci_plots
geci-plot-cli version
```

Then call any command from the table above, substituting your own file names.
