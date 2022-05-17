from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='coap',
    version='1.0.0',
    packages=['coap'],
    url='https://neuralbodies.github.io/COAP',
    license='MIT',
    author='Marko Mihajlovic',
    author_email='markomih@inf.ethz.ch',
    description='COAP body model.',
    long_description=long_description,
    python_requires='>=3.6.0',
    install_requires=[
        'torch>=1.0',
        'numpy>=1.12.2',
        'trimesh',
        'scikit-image',
        'smplx',
    ],
)