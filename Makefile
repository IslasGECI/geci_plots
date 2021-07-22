all: mutants

.PHONY: \
	all \
	check \
	clean \
	coverage \
	format \
	install \
	linter \
	mutants \
	setup \
	tests

module = geci_plots
codecov_token = ff0e4c6d-f104-4657-ba1e-80fd9d1d33a0

define lint
	pylint \
        --disable=bad-continuation \
        --disable=missing-class-docstring \
        --disable=missing-function-docstring \
        --disable=missing-module-docstring \
		--good-names=_,ax,df,i,j,k,kw,n,p,x,y \
        ${1}
endef

check:
	black --check --line-length 100 ${module}
	black --check --line-length 100 setup.py
	black --check --line-length 100 tests
	flake8 --max-line-length 100 ${module}
	flake8 --max-line-length 100 setup.py
	flake8 --max-line-length 100 tests

clean:
	rm --force .mutmut-cache
	rm --recursive --force ${module}.egg-info
	rm --recursive --force ${module}/__pycache__
	rm --recursive --force tests/__pycache__
	rm --recursive --force tests/baseline/
	rm --force --recursive geci_plots/__pycache__/


coverage: setup
	pytest --cov=${module} --cov-report=xml --verbose && \
	codecov --token=${codecov_token}

format:
	black --line-length 100 ${module}
	black --line-length 100 setup.py
	black --line-length 100 tests

install:
	pip install --editable .

linter:
	$(call lint, ${module})
	$(call lint, tests)

mutants: setup
	mutmut run --paths-to-mutate ${module}

setup: install
	mkdir --parents tests/baseline
	pytest --mpl-generate-path tests/baseline/

tests:
	pytest --mpl --verbose
