FROM freqtradeorg/freqtrade:develop

# Install dependencies
COPY requirements-plot.txt /freqtrade/

RUN pip install -r requirements-plot.txt --no-cache-dir

# Empty the ENTRYPOINT to allow all commands
ENTRYPOINT []
