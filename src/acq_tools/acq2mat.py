'''
acq2mat.py

Convert an ACQ file collected via Biopac's AcqKnowledge software to a MATLAB
structure.

The conversion pipeline lives in convert_acq.convert_acq(); this script is a
thin command-line front end over it (the GUI and any programmatic callers use
the same function, so all entry points share one pipeline).
'''

import sys
import argparse
import os

# Dual import: package-relative first, flat fallback for `python acq2mat.py ...`.
try:
    from .convert_acq import convert_acq
except ImportError:
    from convert_acq import convert_acq


def argument_parser(argv):
    '''Parse input from the command line'''

    parser = argparse.ArgumentParser(description='ACQ2MAT: a tool to extract ACQ files.')

    parser.add_argument('file',
        help='ACQ file to convert',
        nargs='+')

    parser.add_argument('-o', '--outfile',
        help='Filename for MATLAB file output',
        required=False)

    args = parser.parse_args()

    if not args.outfile:
        args.outfile = args.file[0].replace('.acq', '.mat')

    return args


def main(argv=None):
    '''Console-script entry point: convert ACQ file(s) to a .mat file.'''
    args = argument_parser(sys.argv[1:] if argv is None else argv)

    # convert_acq() takes (folder, filename); the CLI exposes a single -o path.
    # Split it back apart, mapping an empty directory to the current directory.
    output_folder, output_filename = os.path.split(args.outfile)
    output_folder = output_folder or '.'

    try:
        result = convert_acq(args.file, output_folder, output_filename, fmt='mat')
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    for msg in result.messages:
        print(msg)
    print(f"MAT file saved to: {result.output_path}")


if __name__ == '__main__':
    main()
