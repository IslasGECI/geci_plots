FROM python:3
WORKDIR /workdir
COPY . .
RUN pip install \
    black \
    codecov \
    flake8 \
    mutmut \
    parser \
    pylint \
    pytest \
    pytest-cov \
    pytest-mpl
