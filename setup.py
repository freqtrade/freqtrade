from setuptools import setup

from freqtrade import __version__


setup(name='freqtrade',
      version=__version__,
      description='Simple High Frequency Trading Bot for crypto currencies',
      url='https://github.com/gcarq/freqtrade',
      author='gcarq and contributors',
      author_email='michael.egger@tsn.at',
      license='GPLv3',
      packages=['freqtrade'],
      scripts=['bin/freqtrade'],
      zip_safe=False)
