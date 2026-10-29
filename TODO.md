# Backlog

- [ ] Install the package in the published Docker image. `Dockerfile` installs test tooling only, so `docker run islasgeci/geci_plots:latest geci-plot-cli ...` fails and `README.md` has to work around it with `make install`.
- [ ] Move the Codecov token out of `Makefile`. `codecov_token` is hardcoded in the repository (introduced in `2f2aa8a`); pass it from CI as a secret instead.
- [ ] Extract hardcoded species color dictionary as a input argument `plot_percentage_bar_by_columns()` function. The species color configuration should be a JSON config file input on CLI command `render-species-percentage-coverage-by-quadrant`.
