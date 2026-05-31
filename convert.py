'''
convert.py

In-memory converters between the two ACQ-derived output formats:
  - .mat  (Classic) produced by acq2mat.py  -- nested d.<channel>.{wave,Fs,unit}
  - .h5   (New)      produced by acq2h5.py   -- flat /signals/<name> + attrs

These operate purely on already-converted files; they do NOT require the
original .acq files or the bioread package. The content of the two formats is
identical, so conversion is a faithful structural remap (no recomputation).

Typical batch usage (write your own loop):

    from pathlib import Path
    from convert import mat_to_h5

    for mat_path in Path('data').glob('*.mat'):
        mat_to_h5(mat_path, mat_path.with_suffix('.h5'))

The h5 writer is shared with acq2h5.py (write_h5) so the on-disk .h5 layout
stays identical regardless of whether a file came from an .acq or a .mat.
'''

import os
import json

import numpy as np
import scipy.io as sio
import h5py

from acq2h5 import write_h5, _NON_SIGNAL_KEYS

# Fields stored at the top level of the .mat 'd' struct that are NOT channels.
# Mirrors _NON_SIGNAL_KEYS from acq2h5 (kept in sync via that import).
_MAT_NON_CHANNEL = _NON_SIGNAL_KEYS


def _mat_scalar_str(value):
    '''Extract a Python str from a scipy.io.loadmat string field.'''
    arr = np.ravel(value)
    return str(arr[0]) if arr.size else ''


def _mat_field_to_list(value):
    '''Flatten a loadmat event-marker column into a plain Python list.

    scipy stores these as object/numeric arrays of shape (n,) or (n, 1);
    NaNs are preserved, strings come back as numpy str_ and are coerced to str.
    '''
    out = []
    for v in np.ravel(value):
        if isinstance(v, np.generic):
            v = v.item()
        out.append(v)
    return out


def _read_mat_dict(mat_path):
    '''Load a .mat created by acq2mat.py into the in-memory `d` dict shape that
    write_h5() expects (channels as {'wave','Fs','unit'}, plus event_markers,
    recording_start_*, Fs, timestamps_local).
    '''
    m = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    if 'd' not in m:
        raise ValueError(
            f"{mat_path}: not an acq2mat.py file (no top-level 'd' struct)."
        )
    mat = m['d'][0, 0]
    fields = list(mat._fieldnames)

    d = {}

    # Channels: every field that is not bookkeeping is a nested {wave,Fs,unit}.
    for name in fields:
        if name in _MAT_NON_CHANNEL:
            continue
        ch = getattr(mat, name)[0, 0]
        d[name] = {
            'wave': np.asarray(ch.wave, dtype=np.float64).ravel(),
            'Fs': float(np.ravel(ch.Fs)[0]),
            'unit': _mat_scalar_str(ch.unit),
        }

    # Event markers: struct of parallel arrays -> dict of lists.
    em_struct = getattr(mat, 'event_markers')[0, 0]
    event_markers = {f: _mat_field_to_list(getattr(em_struct, f))
                     for f in em_struct._fieldnames}
    d['event_markers'] = event_markers

    # Top-level scalars / vectors.
    d['recording_start_utc'] = _mat_scalar_str(getattr(mat, 'recording_start_utc'))
    d['recording_start_local'] = _mat_scalar_str(getattr(mat, 'recording_start_local'))
    d['Fs'] = float(np.ravel(getattr(mat, 'Fs'))[0])
    d['timestamps_local'] = np.asarray(
        getattr(mat, 'timestamps_local'), dtype=np.float64).ravel()

    return d


def mat_to_h5(mat_path, h5_path=None):
    '''Convert a .mat (acq2mat.py) file to a flat .h5 (acq2h5.py) file.

    Args:
        mat_path: path to the input .mat file.
        h5_path: output .h5 path. Defaults to mat_path with a .h5 extension.

    Returns:
        The output .h5 path (str).
    '''
    mat_path = os.fspath(mat_path)
    if h5_path is None:
        h5_path = os.path.splitext(mat_path)[0] + '.h5'
    h5_path = os.fspath(h5_path)

    d = _read_mat_dict(mat_path)
    # write_h5 expects event_markers values to be JSON-serializable; the .mat
    # already stores date_created_utc as ISO strings, so no datetime conversion
    # is needed here.
    write_h5(d, h5_path)
    return h5_path


def _h5_attr_str(value):
    '''Coerce an h5py string attribute (str or bytes) to a Python str.'''
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def _h5_scalar(value):
    '''Read an h5py scalar dataset value as a clean Python str/float.'''
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.generic):
        return value.item()
    return value


def h5_to_mat(h5_path, mat_path=None):
    '''Convert a flat .h5 (acq2h5.py) file to a .mat (acq2mat.py) file.

    Produces the same nested d.<channel>.{wave,Fs,unit} layout acq2mat.py writes,
    so existing MATLAB tooling (loadfile.m) keeps working.

    Args:
        h5_path: path to the input .h5 file.
        mat_path: output .mat path. Defaults to h5_path with a .mat extension.

    Returns:
        The output .mat path (str).
    '''
    h5_path = os.fspath(h5_path)
    if mat_path is None:
        mat_path = os.path.splitext(h5_path)[0] + '.mat'
    mat_path = os.fspath(mat_path)

    d = {}

    with h5py.File(h5_path, 'r') as f:
        if 'signals' not in f:
            raise ValueError(
                f"{h5_path}: not an acq2h5.py file (no '/signals' group)."
            )
        signals = f['signals']

        # Channels: every signal dataset except timestamps_local becomes a
        # nested {wave, Fs, unit} struct, matching acq2mat.py output.
        for name in signals.keys():
            if name == 'timestamps_local':
                continue
            ds = signals[name]
            d[name] = {
                'wave': np.asarray(ds, dtype=np.float64),
                'Fs': float(ds.attrs['Fs']) if 'Fs' in ds.attrs else float(f['Fs'][()]),
                'unit': _h5_attr_str(ds.attrs.get('units', '')),
            }

        # event_markers: JSON string -> dict of lists (same shape acq2mat.py
        # builds before savemat).
        event_markers = json.loads(_h5_scalar(f['event_markers'][()]))
        d['event_markers'] = event_markers

        # Top-level metadata.
        d['recording_start_utc'] = _h5_scalar(f['recording_start_utc'][()])
        d['recording_start_local'] = _h5_scalar(f['recording_start_local'][()])
        d['Fs'] = float(f['Fs'][()])
        d['timestamps_local'] = np.asarray(signals['timestamps_local'], dtype=np.float64)

    sio.savemat(mat_path, {'d': d}, oned_as='column', do_compression=True)
    return mat_path
