if ! [[ -v CONDA_PREFIX  ]]; then
    echo "The conda environment is not activated."
    exit 1
fi

INSTALL_LOC=${CONDA_PREFIX}
echo "Installing to ${INSTALL_LOC}"
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
make install || make install
cd .. && rm -rf ./ta-lib/
