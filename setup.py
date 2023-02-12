from setuptools import setup


# Requirements used for submodules
plot = ['plotly>=4.0']
hyperopt = [
    'scipy',
    'scikit-learn',
    'scikit-optimize>=0.7.0',
    'filelock',
    'progressbar2',
]

freqai = [
    'scikit-learn',
    'catboost; platform_machine != "aarch64"',
    'lightgbm',
    'xgboost'
]

freqai_rl = [
    'torch',
    'stable-baselines3',
    'gym==0.21',
    'sb3-contrib'
]

hdf5 = [
    'tables',
    'blosc',
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

all_extra = plot + develop + jupyter + hyperopt + hdf5 + freqai + freqai_rl

setup(
    tests_require=[
        'pytest',
        'pytest-asyncio',
        'pytest-cov',
        'pytest-mock',
    ],
    install_requires=[
        # from requirements.txt
        'ccxt>=2.6.26',
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
        'joblib>=1.2.0',
        'pyarrow; platform_machine != "armv7l"',
        'fastapi',
        'pydantic>=1.8.0',
        'uvicorn',
        'psutil',
        'pyjwt',
        'aiofiles',
        'schedule',
        'websockets',
        'janus'
    ],
    extras_require={
        'dev': all_extra,
        'plot': plot,
        'jupyter': jupyter,
        'hyperopt': hyperopt,
        'hdf5': hdf5,
        'freqai': freqai,
        'freqai_rl': freqai_rl,
        'all': all_extra,
    },
)
