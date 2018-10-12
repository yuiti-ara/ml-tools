from setuptools import setup, find_packages


def run_setup():
    setup(
        name='ml-tools',
        version='0.1',
        author='Yuiti Ara',
        author_email='yuiti.usp@gmail.com',
        license='MIT',
        packages=find_packages(),
    )


if __name__ == '__main__':
    run_setup()
