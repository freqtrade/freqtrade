FROM python:3.12-slim

# Install TA-Lib C library
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make wget && \
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && ./configure --prefix=/usr/local && make -j$(nproc) install 2>/dev/null; \
    ldconfig && \
    cd / && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    apt-get purge -y gcc g++ make wget && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install freqtrade
RUN pip install --no-cache-dir freqtrade

WORKDIR /freqtrade

# Copy strategy and config
COPY TrendRiderStrategy.py /freqtrade/user_data/strategies/
COPY config_mexc.json /freqtrade/

# Create user_data dirs
RUN freqtrade create-userdir --userdir /freqtrade/user_data 2>/dev/null || true

# Railway injects env vars at runtime. Freqtrade reads them with FREQTRADE__ prefix:
# FREQTRADE__EXCHANGE__KEY=your_mexc_key
# FREQTRADE__EXCHANGE__SECRET=your_mexc_secret
# FREQTRADE__DRY_RUN=false  (to go live)

CMD ["freqtrade", "trade", "--config", "/freqtrade/config_mexc.json", "--strategy", "TrendRiderStrategy", "--userdir", "/freqtrade/user_data"]
