from setuptools import setup


# Requirements used for submodules
plot = ['plotly>=4.0']
hyperopt = [
    'scipy',
    'scikit-learn',
    'scikit-optimize>=0.7.0',
    'filelock',
    'joblib',
    'progressbar2',
]

freqai = [
    'scikit-learn',
    'joblib',
    'catboost; platform_machine != "aarch64"',
    'lightgbm',
]

develop = [
    'coveralls',
    'flake8',
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

all_extra = plot + develop + jupyter + hyperopt + freqai

setup(
    tests_require=[
        'pytest',
        'pytest-asyncio',
        'pytest-cov',
        'pytest-mock',
    ],
    install_requires=[
        # from requirements.txt
        'ccxt>=1.92.9',
        'SQLAlchemy',
        'python-telegram-bot>=13.4',
        'arrow>=0.17.0',
        'cachetools',
        'requests',
        'urllib3',
        'jsonschema',
        'TA-Lib',
        'pandas-ta',
        'technical',
        'tabulate',
        'pycoingecko',
        'py_find_1st',
        'python-rapidjson',
        'orjson',
        'sdnotify',
        'colorama',
        'jinja2',
        'questionary',
        'prompt-toolkit',
        'numpy',
        'pandas',
        'tables',
        'blosc',
        'fastapi',
        'uvicorn',
        'psutil',
        'pyjwt',
        'aiofiles',
        'schedule'
    ],
    extras_require={
        'dev': all_extra,
        'plot': plot,
        'jupyter': jupyter,
        'hyperopt': hyperopt,
        'freqai': freqai,
        'all': all_extra,
    },
)
