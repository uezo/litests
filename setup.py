from setuptools import setup, find_packages

setup(
    name="litests",
    version="0.1.0",
    url="https://github.com/uezo/litests",
    author="uezo",
    author_email="uezo@uezo.net",
    maintainer="uezo",
    maintainer_email="uezo@uezo.net",
    description="A super lightweight Speech-to-Speech framework with modular VAD, STT, LLM and TTS components. 🧩",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*"]),
    install_requires=["httpx==0.28.0", "openai==1.55.3", "PyAudio==0.2.14"],
    license="Apache v2",
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)
