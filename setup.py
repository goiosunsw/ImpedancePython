
from setuptools import setup

setup(name='pympedance',
      version='0.1',
      description='Tools to synthesise and measure acoustic impedance',
      url='http://github.com/goiosunw/ImpedancePython',
      author='Andre Goios',
      author_email='a.almeida@unsw.edu.au',
      license='GPL v3',
      packages=['pympedance'],
      install_requires=[
          'xmltodict'
      ],
      zip_safe=False)
