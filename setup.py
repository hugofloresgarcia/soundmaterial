from setuptools import find_packages
from setuptools import setup


setup(
    name="soundmaterial",
    version="0.0.1",
    description="audio data management for vampnet.",
    long_description_content_type="text/markdown",
    author="Hugo Flores García",
    author_email="huferflo@gmail.com",
    url="https://github.com/hugofloresgarcia/soundmaterial",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "pandas", 
        "argbind",
        "gradio", 
        "torchaudio",
        "einops", 
        "julius", 
        "soundfile", 
        "flatten_dict", 
        "scipy", 
        "librosa", 
        "sqlite_web", 
        "pyloudnorm",
    ],
)
