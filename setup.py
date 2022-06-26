
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

all_extra = plot + develop + jupyter + hyperopt

setup(
    tests_require=[
        'pytest',
        'pytest-asyncio',
        'pytest-cov',
        'pytest-mock',
        ],
    install_requires=[
        # from requirements.txt
        'ccxt>=1.50.48',
        'SQLAlchemy',
        'python-telegram-bot>=13.4',
        'arrow>=0.17.0',
        'cachetools',
        'requests',
        'urllib3',
        'wrapt',
        'jsonschema',
        'TA-Lib',
        'technical',
        'tabulate',
        'pycoingecko',
        'py_find_1st',
        'python-rapidjson',
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
        'pyjwt',
        'aiofiles'
    ],
    extras_require={
        'dev': all_extra,
        'plot': plot,
        'jupyter': jupyter,
        'hyperopt': hyperopt,
        'all': all_extra,
    },
)
