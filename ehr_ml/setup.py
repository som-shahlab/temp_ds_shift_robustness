# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['ehr_ml', 'ehr_ml.clmbr', 'ehr_ml.patient2vec']

package_data = \
{'': ['*'], 'ehr_ml': ['extension/*']}

install_requires = \
['embedding-dot @ git+https://github.com/Lalaland/embedding_dot.git@master',
 'numpy==1.21',
 'scikit-learn>=0.24,<0.25',
 'scipy>=1.6,<2.0',
 'torch>=1.10,<2.0',
 'tqdm>=4.60.0,<5.0.0']

entry_points = \
{'console_scripts': ['clmbr_create_info = ehr_ml.clmbr:create_info_program',
                     'clmbr_debug_model = ehr_ml.clmbr:debug_model',
                     'clmbr_train_model = ehr_ml.clmbr:train_model',
                     'ehr_ml_clean_synpuf = ehr_ml.synpuf:clean_synpuf',
                     'ehr_ml_extract_omop = '
                     'ehr_ml.extract:extract_omop_program',
                     'ehr_ml_subset_extract = '
                     'ehr_ml.subset:extract_subset_program',
                     'inspect_timelines = ehr_ml.timeline:inspect_timelines',
                     'patient2vec_debug_model = ehr_ml.patient2vec:debug_model',
                     'patient2vec_train_model = '
                     'ehr_ml.patient2vec:train_model']}

setup_kwargs = {
    'name': 'ehr-ml',
    'version': '0.1.0',
    'description': '',
    'long_description': None,
    'author': 'Ethan Steinberg',
    'author_email': 'ethan.steinberg@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.8,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
