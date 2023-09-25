from setuptools import find_packages, setup

def get_requirements(filename):
    ''' This function will return a list of modules'''
    requirements=[]
    with open(filename, 'r') as file:
        requirements = file.readlines()
        requirements = [module.replace('\n', '') for module in requirements]

    if '-e .' in requirements:
        requirements.remove('-e .')    
    return requirements  
  

setup(
    name='health_insurance_ml_project',
    version=1.0,
    author='Mounica Narnindi',
    packages=find_packages(),
    #install_requires=['numpy', 'pandas']
    install_requires=get_requirements('requirements.txt')
)