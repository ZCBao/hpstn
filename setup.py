from setuptools import setup

setup(
    name='multi_robot_nav',
    version='1.0',
    author='Zhou Chuanbao',
    description="Source code for multi-robot navigation",
    packages=['env', 'policy'],
    install_requires=[
        'gym>=0.11.0,<=0.23.1',
        'imageio',
        "imageio-ffmpeg",
        'matplotlib>=3.0.1,<=3.6.3',
        'numpy',
        'pyyaml',
        'scipy',
        'seaborn>=0.12.2',
        'torchscale',
        'tqdm',
    ],
)