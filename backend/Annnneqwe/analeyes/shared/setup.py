from setuptools import find_packages, setup

setup(
    name='analeyes-shared',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'PyYAML>=6.0',
        'python-dotenv>=1.0',
        'sqlalchemy[asyncio]>=2.0',
        'asyncpg>=0.29',
        'psycopg2-binary>=2.9',
        'alembic>=1.13',
        'pika>=1.3',
        'numpy<2.0',
    ],
)
