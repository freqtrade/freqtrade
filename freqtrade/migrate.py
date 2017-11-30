"""
This script manages the migrations
"""
from pathlib import Path
import os
import caribou

DB_NAME = 'trades'
DB_PATH = '{}.sqlite'.format(DB_NAME)

PARENT_DIR = os.path.abspath('.')
MIGRATIONS_PATH = PARENT_DIR + '/migrations/'
INITIAL_MIGRATION = '20171111135341'

def run():
    """
    Run Migrations
    """
    # Check if old version of db is present
    db_path_v2 = PARENT_DIR + '/{}v2.sqlite'.format(DB_NAME)
    db_path_v3 = PARENT_DIR + '/{}v3.sqlite'.format(DB_NAME)

    # If we use old db file format v2 or v3, rename it to new db filename
    if Path(db_path_v2).is_file():
        os.rename(db_path_v2, DB_PATH)
        # Executes initial migration (catching up)
        caribou.upgrade(DB_PATH, MIGRATIONS_PATH, INITIAL_MIGRATION)
    elif Path(db_path_v3).is_file():
        os.rename(db_path_v3, DB_PATH)
    else:
        # Rename initial migration so it is not executed for next releases
        files = [i for i in os.listdir(MIGRATIONS_PATH) if \
            os.path.isfile(os.path.join(MIGRATIONS_PATH, i)) and \
            INITIAL_MIGRATION in i and \
            '.{}'.format(INITIAL_MIGRATION) not in i]
        if files:
            os.rename('{}{}'.format(MIGRATIONS_PATH, files[0]), \
                '{}.{}'.format(MIGRATIONS_PATH, files[0]))

    # If we have new db file (so from previous renaming or for the future)
    if Path(DB_PATH).is_file():
        # Execute migrations
        caribou.upgrade(DB_PATH, MIGRATIONS_PATH)
