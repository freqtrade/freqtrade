if [ ! -f "ta-lib/CHANGELOG.TXT" ]; then
  tar zxvf ta-lib-0.4.0-src.tar.gz
  cd ta-lib && sed -i "s|0.00000001|0.000000000000000001 |g" src/ta_func/ta_utility.h && ./configure && make && sudo make install && cd ..
else
  echo "TA-lib already installed, skipping download and build."
  cd ta-lib && sed -i "s|0.00000001|0.000000000000000001 |g" src/ta_func/ta_utility.h && sudo make install && cd ..
fi
