from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='MAPS',
    url='https://github.com/NihanPol/MAPS',
    author='Nihan Pol',
    author_email='nihan.pol@nanograv.org',
    # Needed to actually package something
    packages=['MAPS'],
    # Needed for dependencies
    install_requires=['numpy', 'scipy', 'sympy', 'astroML', 'enterprise-pulsar', 'PTMCMCSampler', 'healpy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Package to generate sky maps for PTA stochastic gravitational wave backgrounds.',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)
