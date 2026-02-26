FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make wget libpq-dev && \
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && ./configure --prefix=/usr/local && make -j$(nproc) install 2>/dev/null; \
    ldconfig && \
    cd / && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    apt-get purge -y gcc g++ make wget && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir freqtrade psycopg2-binary

WORKDIR /freqtrade

COPY TrendRider5mStrategy.py /freqtrade/user_data/strategies/
COPY config_5m.json /freqtrade/
COPY start.sh /freqtrade/

RUN freqtrade create-userdir --userdir /freqtrade/user_data 2>/dev/null || true
RUN freqtrade install-ui
RUN chmod +x /freqtrade/start.sh

CMD ["/freqtrade/start.sh"]
