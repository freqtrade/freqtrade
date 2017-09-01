FROM    python:3.6.2

RUN     mkdir -p /freqtrade
WORKDIR /freqtrade

ADD     ./requirements.txt /freqtrade/requirements.txt
RUN     pip install -r requirements.txt
ADD     . /freqtrade
CMD     python main.py
