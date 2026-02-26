FROM python:3.12-slim

# Install TA-Lib + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make wget && \
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && ./configure --prefix=/usr/local && make -j$(nproc) install 2>/dev/null; \
    ldconfig && \
    cd / && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    apt-get purge -y gcc g++ make wget && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir freqtrade

WORKDIR /freqtrade

COPY TrendRiderStrategy.py /freqtrade/user_data/strategies/
COPY config_backtest.json /freqtrade/
COPY download_data.sh /freqtrade/

RUN freqtrade create-userdir --userdir /freqtrade/user_data 2>/dev/null || true
RUN freqtrade install-ui
RUN chmod +x /freqtrade/download_data.sh

CMD ["/freqtrade/download_data.sh"]
