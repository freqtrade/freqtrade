FROM freqtradeorg/freqtrade:develop

# Install dependencies
COPY requirements-dev.txt /freqtrade/

RUN pip install numpy --user --no-cache-dir \
  && pip install -r requirements-dev.txt --user --no-cache-dir

# Empty the ENTRYPOINT to allow all commands
ENTRYPOINT []
