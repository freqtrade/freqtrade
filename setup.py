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
      url='https://github.com/gcarq/freqtrade',
      author='gcarq and contributors',
      author_email='michael.egger@tsn.at',
      license='GPLv3',
      packages=['freqtrade'],
      scripts=['bin/freqtrade'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'pytest-mock', 'pytest-cov'],
      install_requires=[
          'python-bittrex==0.1.3',
          'SQLAlchemy==1.1.13',
          'python-telegram-bot==8.1.1',
          'arrow==0.10.0',
          'requests==2.18.4',
          'urllib3==1.22',
          'wrapt==1.10.11',
          'pandas==0.20.3',
          'scikit-learn==0.19.0',
          'scipy==0.19.1',
          'jsonschema==2.6.0',
          'TA-Lib==0.4.10',
          'tabulate==0.8.1',
      ],
      dependency_links=[
          "git+https://github.com/ericsomdahl/python-bittrex.git@d7033d0#egg=python-bittrex-0.1.3"
      ],
      include_package_data=True,
      zip_safe=False,
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Topic :: Office/Business :: Financial :: Investment',
          'Intended Audience :: Science/Research',
      ])
