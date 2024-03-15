from setuptools import setup, find_packages

setup(
    name='fake_real_news_detection',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'nltk',
        'joblib',
    ],
    entry_points={
        'console_scripts': [
            'run_model=fake_real_news_detection.model:main',
        ],
    },
    author='Artem Reshetnikov',
    author_email='a.reflesh@gmail.com',
    description='A machine learning project to detect fake and real news',
)
