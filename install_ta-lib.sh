if [ ! -f "ta-lib/CHANGELOG.TXT" ]; then
  curl -O -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
  tar zxvf ta-lib-0.4.0-src.tar.gz
  cd ta-lib && ./configure && make && sudo make install && cd ..
else
  echo "TA-lib already installed, skipping download and build."
  cd ta-lib && sudo make install && cd ..
fi
