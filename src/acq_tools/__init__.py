'''
acq-tools: convert and analyze BIOPAC AcqKnowledge physiologic data.

This package converts ACQ files (via the bioread library) into MATLAB (.mat)
and HDF5 (.h5) structures used by MCIRCC research tools, and converts between
those two formats.

When this directory is used as a package inside a larger project (e.g. nested
as `acq_tools/`), import the public API from the package:

    from acq_tools import convert_acq
    result = convert_acq(['rec.acq'], '/out', 'rec', fmt='h5')

The same modules also run as standalone command-line tools from inside this
folder (python acq2mat.py ...), so the internal imports use a package-relative
form with a flat fallback.
'''

# Re-export the most-used entry points so callers can `from acq_tools import X`.
# Guarded so that importing the package never hard-fails if an optional
# dependency for a submodule is missing in a given environment.
try:
    from .convert_acq import convert_acq, ConversionResult, default_filename, FORMATS
    from .acq_common import parse_data, cat_multiple_files, clean
    from .acq2h5 import write_h5
    from .convert import mat_to_h5, h5_to_mat
except ImportError:  # pragma: no cover - allows partial environments
    pass

__all__ = [
    'convert_acq',
    'ConversionResult',
    'default_filename',
    'FORMATS',
    'parse_data',
    'cat_multiple_files',
    'clean',
    'write_h5',
    'mat_to_h5',
    'h5_to_mat',
]
