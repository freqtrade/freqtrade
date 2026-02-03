ARG sourceimage=freqtradeorg/freqtrade
ARG sourcetag=develop
FROM ${sourceimage}:${sourcetag}

# Install dependencies
COPY requirements-freqai.txt /freqtrade/

RUN pip install -r requirements-freqai.txt --user --no-cache-dir
