from sys import version_info
from setuptools import setup

if version_info.major == 3 and version_info.minor < 6 or \
        version_info.major < 3:
    print('Your Python interpreter must be 3.6 or greater!')
    exit(1)

from freqtrade import __version__


setup(name='freqtrade',
      version=__version__,
      description='Simple High Frequency Trading Bot for crypto currencies',
      url='https://github.com/freqtrade/freqtrade',
      author='gcarq and contributors',
      author_email='michael.egger@tsn.at',
      license='GPLv3',
      packages=['freqtrade'],
      scripts=['bin/freqtrade'],
      setup_requires=['pytest-runner', 'numpy'],
      tests_require=['pytest', 'pytest-mock', 'pytest-cov'],
      install_requires=[
          'ccxt',
          'SQLAlchemy',
          'python-telegram-bot',
          'arrow',
          'requests',
          'urllib3',
          'wrapt',
          'pandas',
          'scikit-learn',
          'scipy',
          'joblib',
          'jsonschema',
          'TA-Lib',
          'tabulate',
          'cachetools',
          'coinmarketcap',
          'scikit-optimize',
      ],
      include_package_data=True,
      zip_safe=False,
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Topic :: Office/Business :: Financial :: Investment',
          'Intended Audience :: Science/Research',
      ])
