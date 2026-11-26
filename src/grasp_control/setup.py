import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'grasp_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='jeho4557@gmail.com',
    description='OpenMANIPULATOR-X Grasp and Motion Control Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_node = grasp_control.control_node:main',
            'grasp_router_node = grasp_control.grasp_router_node:main',
            'rb3_dummy_executor_node = grasp_control.rb3_dummy_executor_node:main',
        ],
    },
)
