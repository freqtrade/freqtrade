if [ -z "$1" ]; then
  INSTALL_LOC=/usr/local
else
  INSTALL_LOC=${1}
fi
echo "Installing to ${INSTALL_LOC}"
if [ -n "$2" ] || [ ! -f "${INSTALL_LOC}/lib/libta_lib.a" ]; then
  tar zxvf ta-lib-0.4.0-src.tar.gz
  cd ta-lib \
  && sed -i.bak "s|0.00000001|0.000000000000000001 |g" src/ta_func/ta_utility.h \
  && curl 'http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD' -o config.guess \
  && curl 'http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD' -o config.sub \
  && ./configure --prefix=${INSTALL_LOC}/ \
  && make
  if [ $? -ne 0 ]; then
    echo "Failed building ta-lib."
    cd .. && rm -rf ./ta-lib/
    exit 1
  fi
  if [ -z "$2" ]; then
    which sudo && sudo make install || make install
    if [ -x "$(command -v apt-get)" ]; then
      echo "Updating library path using ldconfig"
      sudo ldconfig
    fi
  else
    # Don't install with sudo
    make install
  fi

  cd .. && rm -rf ./ta-lib/
else
  echo "TA-lib already installed, skipping installation"
fi
