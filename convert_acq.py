'''
convert_acq.py

Reusable, in-process ACQ -> .mat / .h5 conversion.

This is the programmatic counterpart to the acq2mat.py / acq2h5.py command-line
scripts and to run_conversion() in acq2mat_gui.py. Where the GUI shells out to
those scripts via subprocess and reports progress into a window, this module
does the same work *in process* by calling the shared acq_common helpers and the
appropriate writer directly, and returns a structured result instead.

Use it from scripts, notebooks, or batch loops:

    from convert_acq import convert_acq

    result = convert_acq(
        acq_files=['/data/rec_01.acq', '/data/rec_02.acq'],
        output_folder='/out',
        output_filename='2025-06-13',   # extension optional; added per format
        fmt='h5',                       # 'mat' or 'h5'
    )
    print(result.output_path)           # /out/2025-06-13.h5

The conversion pipeline (parse -> optional multi-file concatenation -> validate
single sampling rate -> attach metadata -> build per-sample timestamp vector ->
export the event-marker CSV -> write) is identical to the CLI scripts; only the
final writer differs by format. bioread must be importable in the calling
environment (it does the .acq parsing).
'''

import os
from dataclasses import dataclass, field

import bioread
import pytz
from scipy import io as sio

from acq_common import (
    parse_data,
    cat_multiple_files,
    validate_sampling_rates,
    build_timestamp_vector,
    export_event_markers_csv,
)
from acq2h5 import write_h5

# Output-format registry, mirroring FORMATS in acq2mat_gui.py. The per-sample
# timestamp vector is named differently by format (the values are identical
# unix epoch seconds): the .mat format historically calls it timestamps_local,
# the .h5 format calls it timestamps_unix.
FORMATS = {
    'mat': {'ext': '.mat', 'ts_key': 'timestamps_local'},
    'h5':  {'ext': '.h5',  'ts_key': 'timestamps_unix'},
}

# Timezone the converter treats as "local" (matches acq2mat.py / acq2h5.py).
_LOCAL_TZ = pytz.timezone('US/Eastern')


def default_filename(output_folder, ext=''):
    '''Auto-generate an output filename from the output folder's basename.

    Example: ('/path/to/2025-10-11', '.mat') -> '2025-10-11.mat'. Pass ext=''
    to get just the base name. Returns '' for an empty folder. This is the
    single source of truth for the converter's auto-naming (the GUI imports it).
    '''
    if not output_folder:
        return ''
    folder_name = os.path.basename(output_folder.rstrip(os.sep))
    return f"{folder_name}{ext}"


@dataclass
class ConversionResult:
    '''Outcome of a convert_acq() call.

    Attributes:
        success: True if the output file was written.
        output_path: Absolute path of the written .mat/.h5 file.
        csv_path: Path of the exported event-marker CSV, or None if it was not
            written (CSV export failed non-fatally).
        messages: Human-readable progress/info lines (gaps detected, dtype
            warnings forwarded from the pipeline are printed, not collected;
            this holds the high-level steps this function reports).
    '''
    success: bool
    output_path: str
    csv_path: str = None
    messages: list = field(default_factory=list)


def convert_acq(acq_files, output_folder, output_filename=None, fmt='mat'):
    '''Convert one or more ACQ files to a .mat or .h5 file, in process.

    Args:
        acq_files: an ACQ file path, or a list/tuple of paths. When more than
            one is given they are concatenated in order (with NaN-filled time
            gaps), exactly as the CLI scripts do.
        output_folder: directory to write the output file (and the event-marker
            CSV) into. Created here is NOT attempted -- it must already exist.
        output_filename: output file name, with or without the format
            extension. If None/empty, defaults to the output folder's basename
            (e.g. folder ".../2025-06-13" -> "2025-06-13<ext>"), matching the
            GUI's auto-naming.
        fmt: 'mat' or 'h5'.

    Returns:
        ConversionResult with the output path and the exported CSV path.

    Raises:
        ValueError: on bad arguments, on a sampling-rate mismatch across
            channels, or when the event-marker CSV cannot be written (a missing
            marker timestamp is a hard error, same as the CLI).
        FileNotFoundError: if an input ACQ file or the output folder is missing.
    '''
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format '{fmt}'. Expected one of {sorted(FORMATS)}.")
    ext = FORMATS[fmt]['ext']
    ts_key = FORMATS[fmt]['ts_key']

    # Normalize the file list (accept a single path or an iterable of paths).
    if isinstance(acq_files, (str, os.PathLike)):
        file_list = [os.fspath(acq_files)]
    else:
        file_list = [os.fspath(f) for f in acq_files if f]
    if not file_list:
        raise ValueError("No ACQ files provided.")

    if not output_folder:
        raise ValueError("No output folder provided.")
    if not os.path.isdir(output_folder):
        raise FileNotFoundError(f"Output folder does not exist: {output_folder}")

    missing = [f for f in file_list if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(f"ACQ file(s) not found: {', '.join(missing)}")

    # Resolve the output filename (auto-name from folder if absent), and ensure
    # it carries the right extension for the chosen format.
    messages = []
    if not output_filename:
        output_filename = default_filename(output_folder, ext)
        messages.append(f"Using auto-generated filename: {output_filename}")
    if not output_filename.endswith(ext):
        output_filename += ext
    output_path = os.path.join(output_folder, output_filename)

    # --- Shared pipeline (identical to acq2mat.py / acq2h5.py __main__) -------

    data = [bioread.read_file(f) for f in file_list]

    parsed = [parse_data(i) for i in data]
    d_list = [p[0] for p in parsed]
    start_times = [p[1] for p in parsed]
    file_meta = parsed[0][2]  # file-level provenance from the first file

    if len(d_list) >= 2:
        d = cat_multiple_files(d_list, start_times)
    else:
        d = d_list[0]

    # The output layouts assume one shared sampling rate (single timestamp axis,
    # single top-level Fs); a mismatch is a hard error.
    validate_sampling_rates(d)

    # Attach time metadata. channel_keys is frozen here, before any top-level
    # bookkeeping keys are added, so they are never mistaken for channels.
    channel_keys = [k for k in d.keys() if k != 'event_markers']
    Fs = d[channel_keys[0]]['Fs']
    start_time_local = start_times[0].astimezone(_LOCAL_TZ)
    d['recording_start_utc'] = start_times[0].isoformat()
    d['recording_start_local'] = start_time_local.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    d['Fs'] = Fs

    # Per-sample timestamp vector (unix epoch seconds), under the format's key.
    n_samples = len(d[channel_keys[0]]['wave'])
    d[ts_key] = build_timestamp_vector(start_time_local, Fs, n_samples)

    # Export event markers to CSV alongside the output file.
    csv_date = start_time_local.strftime('%Y-%m-%d')
    csv_path = os.path.join(output_folder, f'{csv_date}_extracted_comments.csv')
    try:
        export_event_markers_csv(d['event_markers'], csv_path)
        messages.append(f"Event markers exported to: {csv_path}")
    except ValueError:
        # A missing marker timestamp prevents a correct CSV -- hard error, same
        # as the CLI scripts (which sys.exit(1) here).
        raise
    except Exception as e:
        # Any other CSV failure is non-fatal; the conversion still proceeds.
        messages.append(f"Warning: Could not export CSV: {e}")
        csv_path = None

    # Datetimes -> ISO strings for JSON/MAT/HDF5 serialization.
    if 'date_created_utc' in d['event_markers']:
        d['event_markers']['date_created_utc'] = [
            dt.isoformat() for dt in d['event_markers']['date_created_utc']
        ]

    # --- Format-specific writer ----------------------------------------------

    if fmt == 'h5':
        # write_h5 takes file-level provenance separately (root attributes).
        write_h5(d, output_path, file_meta)
    else:
        # .mat stores file-level provenance as top-level fields, then wraps the
        # whole dict in one struct named 'd' (matching acq2mat.py).
        d['file_revision'] = file_meta['file_revision']
        d['native_samples_per_second'] = file_meta['native_samples_per_second']
        sio.savemat(output_path, {'d': d}, oned_as='column', do_compression=True)

    messages.append(f"Output saved to: {output_path}")
    return ConversionResult(
        success=True,
        output_path=output_path,
        csv_path=csv_path,
        messages=messages,
    )
