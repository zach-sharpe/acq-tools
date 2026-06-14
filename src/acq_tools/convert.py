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

# Dual import: package-relative first, flat fallback for standalone execution.
try:
    from .acq2h5 import write_h5, _NON_SIGNAL_KEYS
except ImportError:
    from acq2h5 import write_h5, _NON_SIGNAL_KEYS

# Fields stored at the top level of the .mat 'd' struct that are NOT channels.
# Mostly mirrors acq2h5's _NON_SIGNAL_KEYS, but the .mat format names its
# per-sample timestamp field timestamps_local (the .h5 format renamed it to
# timestamps_unix). Same values, different key -- so swap the name here, or the
# .mat's timestamps_local field gets misread as a channel.
_MAT_NON_CHANNEL = (_NON_SIGNAL_KEYS - {'timestamps_unix'}) | {
    'timestamps_local', 'file_revision', 'native_samples_per_second'}


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
    recording_start_*, Fs, timestamps_unix).
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
        chan = {
            'wave': np.asarray(ch.wave, dtype=np.float64).ravel(),
            'Fs': float(np.ravel(ch.Fs)[0]),
            'unit': _mat_scalar_str(ch.unit),
        }
        # Channel provenance (added alongside the .h5 channel attrs). Older .mat
        # files predating these fields won't have them, so each is optional.
        ch_fields = set(ch._fieldnames)
        if 'order_num' in ch_fields:
            chan['order_num'] = int(np.ravel(ch.order_num)[0])
        if 'point_count' in ch_fields:
            chan['point_count'] = int(np.ravel(ch.point_count)[0])
        if 'frequency_divider' in ch_fields:
            chan['frequency_divider'] = int(np.ravel(ch.frequency_divider)[0])
        if 'raw_scale_factor' in ch_fields:
            chan['raw_scale_factor'] = float(np.ravel(ch.raw_scale_factor)[0])
        if 'raw_offset' in ch_fields:
            chan['raw_offset'] = float(np.ravel(ch.raw_offset)[0])
        if 'dtype' in ch_fields:
            chan['dtype'] = _mat_scalar_str(ch.dtype)
        d[name] = chan

    # Event markers: struct of parallel arrays -> dict of lists.
    em_struct = getattr(mat, 'event_markers')[0, 0]
    event_markers = {f: _mat_field_to_list(getattr(em_struct, f))
                     for f in em_struct._fieldnames}
    d['event_markers'] = event_markers

    # Top-level scalars / vectors.
    d['recording_start_utc'] = _mat_scalar_str(getattr(mat, 'recording_start_utc'))
    d['recording_start_local'] = _mat_scalar_str(getattr(mat, 'recording_start_local'))
    d['Fs'] = float(np.ravel(getattr(mat, 'Fs'))[0])
    # The .mat format names this field timestamps_local; the .h5 format calls it
    # timestamps_unix. The values are identical (unix epoch seconds) -- only the
    # key differs -- so remap to the key write_h5() expects.
    d['timestamps_unix'] = np.asarray(
        getattr(mat, 'timestamps_local'), dtype=np.float64).ravel()

    # File-level provenance (optional; absent in legacy .mat files). Returned
    # separately so mat_to_h5 can forward it to write_h5 as root attributes.
    file_meta = {}
    if 'file_revision' in fields:
        file_meta['file_revision'] = int(np.ravel(getattr(mat, 'file_revision'))[0])
    if 'native_samples_per_second' in fields:
        file_meta['native_samples_per_second'] = float(
            np.ravel(getattr(mat, 'native_samples_per_second'))[0])

    return d, file_meta


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

    d, file_meta = _read_mat_dict(mat_path)
    # write_h5 expects event_markers values to be JSON-serializable; the .mat
    # already stores date_created_utc as ISO strings, so no datetime conversion
    # is needed here. file_meta carries file-level provenance to root attrs.
    write_h5(d, h5_path, file_meta)
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

        # Channels: every signal dataset except timestamps_unix becomes a
        # nested {wave, Fs, unit} struct, matching acq2mat.py output.
        for name in signals.keys():
            if name == 'timestamps_unix':
                continue
            ds = signals[name]
            chan = {
                'wave': np.asarray(ds, dtype=np.float64),
                'Fs': float(ds.attrs['Fs']) if 'Fs' in ds.attrs else float(f['Fs'][()]),
                'unit': _h5_attr_str(ds.attrs.get('units', '')),
            }
            # Channel provenance attrs -> nested struct fields (parity with the
            # .h5 attrs). Each is carried only when present on the dataset.
            if 'order_num' in ds.attrs:
                chan['order_num'] = int(ds.attrs['order_num'])
            if 'point_count' in ds.attrs:
                chan['point_count'] = int(ds.attrs['point_count'])
            if 'frequency_divider' in ds.attrs:
                chan['frequency_divider'] = int(ds.attrs['frequency_divider'])
            if 'raw_scale_factor' in ds.attrs:
                chan['raw_scale_factor'] = float(ds.attrs['raw_scale_factor'])
            if 'raw_offset' in ds.attrs:
                chan['raw_offset'] = float(ds.attrs['raw_offset'])
            if 'dtype' in ds.attrs:
                chan['dtype'] = _h5_attr_str(ds.attrs['dtype'])
            d[name] = chan

        # event_markers: JSON string -> dict of lists (same shape acq2mat.py
        # builds before savemat).
        event_markers = json.loads(_h5_scalar(f['event_markers'][()]))
        d['event_markers'] = event_markers

        # Top-level metadata.
        d['recording_start_utc'] = _h5_scalar(f['recording_start_utc'][()])
        d['recording_start_local'] = _h5_scalar(f['recording_start_local'][()])
        d['Fs'] = float(f['Fs'][()])
        # .h5 stores this as signals/timestamps_unix; the .mat format keeps the
        # historical field name timestamps_local (same unix-second values).
        d['timestamps_local'] = np.asarray(signals['timestamps_unix'], dtype=np.float64)

        # File-level provenance from root attrs -> top-level .mat fields
        # (parity with acq2mat.py). Carried only when present.
        if 'file_revision' in f.attrs:
            d['file_revision'] = int(f.attrs['file_revision'])
        if 'native_samples_per_second' in f.attrs:
            d['native_samples_per_second'] = float(f.attrs['native_samples_per_second'])

    sio.savemat(mat_path, {'d': d}, oned_as='column', do_compression=True)
    return mat_path
