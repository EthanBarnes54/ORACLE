"""Build the corrected TimeTagger4 wrapper, timetagger4, on the rack PC.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

From this folder, with ORACLE's environment:

    uv run --with pybind11 --with setuptools python build_wrapper.py
        [--driver "C:\\Program Files\\cronologic\\TimeTagger4\\driver"]

It compiles timetagger4.cpp against the cronologic driver with the Visual
C++ build tools and leaves timetagger4.<tag>.pyd beside a copy of the
driver's DLL. Point the settings' instruments: tagger_driver here.

Depends on: setuptools, pybind11; Visual Studio's C++ build tools; the
cronologic TimeTagger4 driver.
"""

import argparse
import os
import shutil
import sys

DEFAULT_DRIVER = r'C:\Program Files\cronologic\TimeTagger4\driver'
DRIVER_LIBRARY = 'xtdc4_driver_64'
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Build ORACLE's corrected TimeTagger4 "
                                                 "wrapper.")
    parser.add_argument('--driver', default=DEFAULT_DRIVER,
                        help="the cronologic driver's folder, holding include and x64")
    args = parser.parse_args()
    include, library = os.path.join(args.driver, 'include'), os.path.join(args.driver, 'x64')
    for folder in (include, library):
        if not os.path.isdir(folder):
            sys.exit(f"{folder} does not exist: install the cronologic TimeTagger4 driver, or "
                     f"give its folder with --driver.")
    import pybind11
    from setuptools import Extension, setup
    extension = Extension('timetagger4', [os.path.join(HERE, 'timetagger4.cpp')],
                          include_dirs=[pybind11.get_include(), include],
                          library_dirs=[library], libraries=[DRIVER_LIBRARY], language='c++',
                          extra_compile_args=['/std:c++17', '/EHsc'])
    os.chdir(HERE)
    setup(name='timetagger4', ext_modules=[extension],
          script_args=['build_ext', '--inplace', '--build-temp', os.path.join(HERE, 'build')])
    shutil.copy2(os.path.join(library, f"{DRIVER_LIBRARY}.dll"), HERE)
    print(f"Built in {HERE}; set instruments: tagger_driver to it in the settings.")


if __name__ == '__main__':
    main()
