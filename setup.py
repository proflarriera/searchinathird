from setuptools import setup, find_packages

# Función para leer las dependencias de requirements.txt
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='search_in_a_third',
    version='0.1.1',
    packages=find_packages(),
    description='A Python package for efficient hyperparameter optimization in neural networks, using a greedy algorithm guided by heuristic directions.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Diego Larriera',
    author_email='proflarriera@gmail.com',
    url='https://github.com/proflarriera/searchinathird', 
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords='hyperparameter optimization, neural networks, machine learning, greedy algorithm, heuristic',
    install_requires=read_requirements(),
)