from setuptools import setup, find_packages
import os

description = "lewip_informal: the model for tranfering formal text to informal " \
              " which preserves either predefined or automatically-detected important slots from original text" 

long_description = description
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name='lewip_informal',
    packages = ['lewip_informal'],
    version='0.0.0',
    license='Apache',
    author="Nikolay Babakov",
    author_email='bbkhse@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/s-nlp/LEWIP-informal',
    install_requires=[
          'transformers', 'torch'
      ],
)