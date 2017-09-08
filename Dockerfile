FROM    python:3.6.2

RUN pip install numpy
RUN apt-get update
RUN apt-get -y install build-essential  
RUN  wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
RUN tar zxvf ta-lib-0.4.0-src.tar.gz
RUN cd ta-lib && ./configure && make && make install
ENV LD_LIBRARY_PATH /usr/local/lib

RUN     mkdir -p /freqtrade
WORKDIR /freqtrade

ADD     ./requirements.txt /freqtrade/requirements.txt
RUN     pip install -r requirements.txt
ADD     . /freqtrade
CMD     python main.py
