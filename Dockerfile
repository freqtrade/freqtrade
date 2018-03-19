FROM python:3.6.4-slim-stretch

# Install TA-lib
RUN apt-get update && apt-get -y install curl build-essential && apt-get clean
RUN curl -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz | \
  tar xzvf - && \
  cd ta-lib && \
  ./configure && make && make install && \
  cd .. && rm -rf ta-lib
ENV LD_LIBRARY_PATH /usr/local/lib

# Prepare environment
RUN mkdir /freqtrade
WORKDIR /freqtrade

# Install dependencies
COPY requirements.txt /freqtrade/
RUN pip install -r requirements.txt

# Install and execute
COPY . /freqtrade/
RUN pip install -e .
ENTRYPOINT ["freqtrade"]
