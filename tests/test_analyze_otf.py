#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

"""" OTF Parsing test suite based on py.test

Steven Torrisi
"""

import os
import sys

sys.path.append('../modules')
from analyze_otf import parse_md_information, parse_dft_information


def test_parse_md():
    os.system('cp test_files/otf_output_1.out otf_run.out')
    _, _, _, _ = parse_md_information('otf_run.out')
    os.system('rm otf_run.out')


def test_parse_dft():
    os.system('cp test_files/otf_output_1.out otf_run.out')
    _, _, _ = parse_dft_information('otf_run.out')
    os.system('rm otf_run.out')
