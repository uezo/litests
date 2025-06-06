from setuptools import setup, find_packages

setup(
    name="litests",
    version="0.3.12",
    url="https://github.com/uezo/litests",
    author="uezo",
    author_email="uezo@uezo.net",
    maintainer="uezo",
    maintainer_email="uezo@uezo.net",
    description="A super lightweight Speech-to-Speech framework with modular VAD, STT, LLM and TTS components. 🧩",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*"]),
    install_requires=["httpx>=0.27.0", "openai>=1.55.3", "aiofiles>=24.1.0"],
    license="Apache v2",
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)
