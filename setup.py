from setuptools import setup, find_packages, Command
import shutil
import os

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Remove build artifacts."""
        print("Cleaning up build artifacts...")
        shutil.rmtree('build', ignore_errors=True)
        shutil.rmtree('dist', ignore_errors=True)
        shutil.rmtree('molstf_problems.egg-info', ignore_errors=True)
        for root, dirs, files in os.walk('.'):
            for filename in files:
                if filename.endswith('.pyc') or filename.endswith('.pyo'):
                    os.remove(os.path.join(root, filename))
            for dirname in dirs:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(root, dirname))

setup(
    name='molstf_problems',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pymoo',
        'matplotlib',
    ],
    include_package_data=True,
    author='Angus Kenny, Tapabrata Ray, and Hemant Kumar Singh',
    author_email='angus.kenny@unsw.edu.au',
    description='Multi-objective L-shaped Test Functions',
    url='https://www.mdolab.net/Ray/Research-Data/molstf_problems.zip',
    cmdclass={
        'clean': CleanCommand,
    },
)