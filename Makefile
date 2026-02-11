all: mutants

.PHONY: \
	all \
	check \
	clean \
	coverage \
	format \
	green \
	init \
	install \
	linter \
	mutants \
	red \
	refactor \
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
        ${1}
endef

check:
	black --check --line-length 100 ${module}
	black --check --line-length 100 tests
	flake8 --max-line-length 100 ${module}
	flake8 --max-line-length 100 tests
	mypy ${module}
	mypy tests

clean:
	rm --force --recursive ${module}.egg-info
	rm --force --recursive ${module}/__pycache__
	rm --force --recursive geci_plots/__pycache__/
	rm --force --recursive tests/__pycache__
	rm --force --recursive tests/baseline/
	rm --force .mutmut-cache


coverage: setup
	pytest --cov=${module} --cov-report=xml --verbose && \
	codecov --token=${codecov_token}

format:
	black --line-length 100 ${module}
	black --line-length 100 tests

init: init_git setup tests

init_git:
	git config --global --add safe.directory /workdir
	git config --global user.name "Ciencia de Datos • GECI"
	git config --global user.email "ciencia.datos@islas.org.mx"

install:
	pip install --editable .

linter:
	$(call lint, ${module})
	$(call lint, tests)

mutants: setup
	mutmut run
	expr "{mutmut results | wc -l}" == "0"

setup: clean install
	mypy --install-types
	mkdir --parents tests/baseline
	pytest --mpl-generate-path tests/baseline/

tests:
	pytest --mpl --verbose tests/

tests_location = tests/
red: format
	pytest --verbose ${tests_location} \
	&& git restore tests/*.py \
	|| (git add tests/*.py && git commit -m "🛑🧪 Fail tests")
	chmod g+w -R .

green: format
	pytest --verbose ${tests_location} \
	&& (git add ${module}/*.py tests/*.py && git commit -m "✅ Pass tests") \
	|| git restore ${module}/*.py
	chmod g+w -R .

refactor: format
	pytest --verbose ${tests_location} \
	&& (git add ${module}/*.py tests/*.py && git commit -m "♻️  Refactor ${message}") \
	|| git restore ${module}/*.py tests/*.py
	chmod g+w -R .
