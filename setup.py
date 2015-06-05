try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Roof detection project',
    'author': 'Angela Santin',
    #'url': 'URL to get it at.',
    #'download_url': 'Where to download it.',
    #'author_email': 'angela.santin@outlook.com',
    'version': '0.1',
    'install_requires': [],
    'packages': [],
    'scripts': [],
    'name': 'Roof detection'
}

setup(**config)
