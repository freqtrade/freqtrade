FROM python:3.7.0-slim-stretch

# Install TA-lib
RUN apt-get update && apt-get -y install curl build-essential && apt-get clean
RUN curl -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz | \
  tar xzvf - && \
  cd ta-lib && \
  sed -i "s|0.00000001|0.000000000000000001 |g" src/ta_func/ta_utility.h && \
  ./configure && make && make install && \
  cd .. && rm -rf ta-lib
ENV LD_LIBRARY_PATH /usr/local/lib

# Prepare environment
RUN mkdir /freqtrade
WORKDIR /freqtrade

# Install dependencies
COPY requirements.txt /freqtrade/
RUN pip install numpy --no-cache-dir \
  && pip install -r requirements.txt --no-cache-dir

# Install and execute
COPY . /freqtrade/
RUN pip install -e . --no-cache-dir
ENTRYPOINT ["freqtrade"]
