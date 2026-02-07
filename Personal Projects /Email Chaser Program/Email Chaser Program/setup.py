from setuptools import setup

APP = ['main.py']
DATA_FILES = ['contactsdb.csv', 'credentials.json']
OPTIONS = {}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
