from setuptools import setup

setup(name='somatotopic_maps',
      version='0.1',
      description='Somatotopic maps: create and analyse somatotopic maps',
      url='https://github.com/lauraredmondson/tactile_maps',
      author='Laura Edmondson Chua',
      author_email='laurarechua@gmail.com',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
      ],
      packages=['somatotopic_maps'],
      zip_safe=False,
      install_requires = [
        'touchsim @ git+ssh://git@github.com:hsaal/touchsim.git'
      ]
)
