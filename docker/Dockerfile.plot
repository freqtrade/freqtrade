ARG sourceimage=develop
FROM freqtradeorg/freqtrade:${sourceimage}

# Install dependencies
COPY requirements-plot.txt /freqtrade/

RUN pip install -r requirements-plot.txt --no-cache-dir
