from setuptools import setup


# Requirements used for submodules
plot = ['plotly>=4.0']
hyperopt = [
    'scipy',
    'scikit-learn<=1.1.3',
    'scikit-optimize>=0.7.0',
    'filelock',
]

freqai = [
    'scikit-learn',
    'joblib',
    'catboost; platform_machine != "aarch64"',
    'lightgbm',
    'xgboost',
    'tensorboard',
    'datasieve>=0.1.5'
]

freqai_rl = [
    'torch',
    'gymnasium',
    'stable-baselines3',
    'sb3-contrib',
    'tqdm'
]

hdf5 = [
    'tables',
    'blosc',
]

develop = [
    'coveralls',
    'mypy',
    'ruff',
    'pre-commit',
    'pytest',
    'pytest-asyncio',
    'pytest-cov',
    'pytest-mock',
    'pytest-random-order',
    'isort',
    'time-machine',
    'types-cachetools',
    'types-filelock',
    'types-requests',
    'types-tabulate',
    'types-python-dateutil'
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
        'ccxt>=3.0.0',
        'SQLAlchemy>=2.0.6',
        'python-telegram-bot>=20.1',
        'arrow>=1.0.0',
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
        'rich',
        'pyarrow; platform_machine != "armv7l"',
        'fastapi',
        'pydantic>=1.8.0',
        'uvicorn',
        'psutil',
        'pyjwt',
        'aiofiles',
        'schedule',
        'websockets',
        'janus',
        'ast-comments',
        'aiohttp',
        'cryptography',
        'httpx>=0.24.1',
        'python-dateutil',
        'packaging',
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
