from setuptools import setup, find_packages

setup(
    name="openo1",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "tqdm",
        # 添加其他依赖
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="OpenO1: 一个基于大型语言模型的开源项目，结合CoT、RL和MCTS提升AI推理能力",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SWORD-ZEUS/openo1",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
