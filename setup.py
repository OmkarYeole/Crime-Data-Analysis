from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_required_packages(file_path : str) -> List[str]:
    '''
    This function will return list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author=['Rishikesh','Vedant','Jason'],
    author_email=['riso2414@colorado.edu','veku9185@colorado.edu','jaab9381@colorado.edu'],
    packages=find_packages(),
    install_requires= get_required_packages('requirements.txt')
)