from sys import version_info
from setuptools import setup

if version_info.major == 3 and version_info.minor < 6 or \
        version_info.major < 3:
    print('Your Python interpreter must be 3.6 or greater!')
    exit(1)

from pathlib import Path  # noqa: E402
from freqtrade import __version__  # noqa: E402


readme_file = Path(__file__).parent / "README.md"
readme_long = "Crypto Trading Bot"
if readme_file.is_file():
    readme_long = (Path(__file__).parent / "README.md").read_text()

# Requirements used for submodules
api = ['flask']
plot = ['plotly>=4.0']
hyperopt = [
    'scipy',
    'scikit-learn',
    'scikit-optimize',
    'filelock',
    'joblib',
    ]

develop = [
    'coveralls',
    'flake8',
    'flake8-type-annotations',
    'flake8-tidy-imports',
    'mypy',
    'pytest',
    'pytest-asyncio',
    'pytest-cov',
    'pytest-mock',
    'pytest-random-order',
]

jupyter = [
    'jupyter',
    'nbstripout',
    'ipykernel',
    'nbconvert',
    ]

all_extra = api + plot + develop + jupyter + hyperopt

setup(name='freqtrade',
      version=__version__,
      description='Crypto Trading Bot',
      long_description=readme_long,
      long_description_content_type="text/markdown",
      url='https://github.com/freqtrade/freqtrade',
      author='Freqtrade Team',
      author_email='michael.egger@tsn.at',
      license='GPLv3',
      packages=['freqtrade'],
      setup_requires=['pytest-runner', 'numpy'],
      tests_require=['pytest', 'pytest-mock', 'pytest-cov'],
      install_requires=[
          # from requirements-common.txt
          'ccxt>=1.18.1080',
          'SQLAlchemy',
          'python-telegram-bot',
          'arrow',
          'cachetools',
          'requests',
          'urllib3',
          'wrapt',
          'jsonschema',
          'TA-Lib',
          'tabulate',
          'coinmarketcap',
          'py_find_1st',
          'python-rapidjson',
          'sdnotify',
          'colorama',
          'jinja2',
          # from requirements.txt
          'numpy',
          'pandas',
      ],
      extras_require={
          'api': api,
          'dev': all_extra,
          'plot': plot,
          'jupyter': jupyter,
          'hyperopt': hyperopt,
          'all': all_extra,
      },
      include_package_data=True,
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'freqtrade = freqtrade.main:main',
          ],
      },
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Topic :: Office/Business :: Financial :: Investment',
          'Intended Audience :: Science/Research',
      ])
