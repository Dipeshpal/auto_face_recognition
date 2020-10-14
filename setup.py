import setuptools
from setuptools import find_namespace_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto_face_recognition",
    version="0.0.1",
    author="Dipesh",
    author_email="dipeshpal17@gmail.com",
    description="auto_face_recognition is Tensorflow based python library for fast face recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dipeshpal/auto_face_recognition",
    include_package_data=True,
    packages=find_namespace_packages(include=['auto_face_recognition.*', 'auto_face_recognition']),
    install_requires=['opencv-contrib-python',
                      'tensorflow', 'matplotlib'],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
