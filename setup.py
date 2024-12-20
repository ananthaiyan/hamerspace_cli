
from setuptools import setup, find_packages

setup(
    name='hamerspace',
    version='0.1.0',
    description='CLI tool for shrinking and optimizing AI models',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'onnx',
        'onnxruntime',
        'tensorflow-model-optimization',
        'click',
    ],
    entry_points={
        'console_scripts': [
            'hamerspace=hamerspace.cli:main',
        ],
    },
)
