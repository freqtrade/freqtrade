FROM    python:3.6.2

RUN apt-get update
RUN apt-get -y install build-essential

# Install TA-lib
RUN  wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
RUN tar zxvf ta-lib-0.4.0-src.tar.gz
RUN cd ta-lib && ./configure && make && make install
ENV LD_LIBRARY_PATH /usr/local/lib

# Prepare environment
RUN mkdir /freqtrade
COPY . /freqtrade/
WORKDIR /freqtrade

# Install dependencies and execute
RUN pip install -r requirements.txt
RUN pip install -e .
CMD ["freqtrade"]
