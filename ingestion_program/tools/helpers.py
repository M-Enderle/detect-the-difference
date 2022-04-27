import os
from pylint import lint
from pylint.reporters.text import TextReporter
import re


class WritableObject():
    """ Dummy output stream for pylint """
    def __init__(self):
        self.content = []

    def write(self, input_string):
        """ Dummy write """
        self.content.append(input_string)

    def read(self):
        """ Dummy read """
        return self.content


def run_pylint(filename, verbose=False):
    """ Run pylint on the given file """
    package_whitelist = 'numpy,torch,cv2'
    pylint_args = ['--max-line-length=120', '--disable=invalid-name', f'--extension-pkg-whitelist={package_whitelist}',
                   f'--ignored-modules={package_whitelist}', f'--ignored-classes={package_whitelist}']
    pylint_output = WritableObject()

    print("Checking Python code quality")

    # creating temporary __init__.py files such that pylint checks all directories and subdirectories
    init_files = []
    for path in [f.path for f in os.scandir(os.path.dirname(filename)) if f.is_dir()] \
                + [os.path.abspath(os.path.dirname(filename))]:
        init_file = os.path.join(path, '__init__.py')
        if not os.path.exists(init_file):
            open(init_file, 'w').close()
            init_files.append(init_file)

    lint.Run([os.path.dirname(filename)]+pylint_args, reporter=TextReporter(pylint_output), do_exit=False)

    for init_file in init_files:
        os.remove(init_file)

    output = list(pylint_output.read())
    if output:
        match = re.match(r'[^\d]+(\-?\d{1,3}\.\d{2}).*', output[-3])
        if not match:
            return 0
        rate = float(match.groups()[0]) * 10

        with open(filename) as file:
            if 'pylint: disable' in file.read() or 'pylint:disable' in file.read():
                print('Your code quality rating has been set to 0 as you used a pylint: ignore flag in your code.')
                rate = 0

    else:
        rate = 100.0

    if verbose:
        print(''.join(pylint_output.read()))

    return rate
