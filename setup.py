from setuptools import setup, find_packages

setup(
    name='deepsynth',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy', 'tensorflow', 'hyperopt', 'matplotlib', 'seaborn'],
    description='A Python package for synthesizing deep learning models with automatic optimization.',
    author='Venom Ayush',
    author_email='venombolteop@example.com',
    license='MIT',
    url='https://github.com/venombolteop/deepsynth',
)