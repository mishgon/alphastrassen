from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='alphastrassen',
    version='0.0.1',
    description='Reproduction of AlphaTensor paper for 2x2 matrices.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/migonch/alphastrassen',
    author='M. Goncharov',
    author_email='Mikhail.Goncharov2@skoltech.ru',
    packages=find_packages(include=('alphastrassen',)),
    python_requires='>=3.6',
    install_requires=requirements,
)
