from setuptools import setup
import io

VERSION = '0.0.1'

def main():
    with io.open("README.md", encoding="utf8") as f:
        long_description = f.read().strip()

    with open("requirements.txt") as f:
            required = f.read().splitlines()

    setup(
        name='srmdpy',
        url='https://github.com/njericha/srmdpy',
        author='Nicholas Richardson',
        author_email='njericha@uwaterloo.ca',
        packages=['srmdpy'],
        install_requires=required,
        version=VERSION,
        license='MIT',
        description='A Python implimentation of Sparse Random Mode Decomposition',
        long_description=long_description,
        long_description_content_type="text/markdown",
    )

if __name__ == '__main__':
    main()