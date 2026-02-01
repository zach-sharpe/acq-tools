'''
acq2mat.py

Convert an ACQ file collected via Biopac's AcqKnowledge software to a MATLAB structure.
'''

import sys
import argparse
import re
import json
import os
from datetime import datetime, timedelta
import bioread
import numpy as np
import pandas as pd
from scipy import io as sio
import pytz
import pdb

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

def clean(s):
    '''Take a string and construct a valid variable name.
    Used to ensure that channel names are able to be used in MATLAB.
    Adapted from https://stackoverflow.com/questions/3303312/how-do-i-convert-a-string-to-a-valid-variable-name-in-python'''

    s = s.lower().strip() # preliminary cleaning
    s = re.sub('[^0-9a-zA-Z_]', '', s) # include only valid characters
    s = re.sub('^[^a-zA-Z]+', '', s) # first character must be a letter

    return s

def load_signal_renaming():
    '''Load signal renaming mappings from JSON file.
    Returns dict of mappings, or empty dict if file doesn't exist.'''

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'signal_renaming.json')

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load signal_renaming.json: {e}")
            return {}
    else:
        return {}

def timestamps_to_datetime(timestamps, tz_name='US/Eastern'):
    '''Convert unix timestamps from the MAT file to Python datetime objects.

    Args:
        timestamps: scalar, list, or numpy array of unix timestamps (seconds since epoch)
        tz_name: pytz timezone string (default: 'US/Eastern')

    Returns:
        list of timezone-aware datetime objects, or a single datetime if
        a scalar was passed
    '''
    tz = pytz.timezone(tz_name)
    scalar = np.isscalar(timestamps)
    timestamps = np.asarray(timestamps, dtype=np.float64).ravel()

    result = [datetime.fromtimestamp(float(ts), tz=tz) for ts in timestamps]

    return result[0] if scalar else result

def validate_sampling_rates(d):
    '''Verify all channels share the same sampling frequency.

    Raises ValueError if any channel has a different Fs.
    '''
    channel_keys = [k for k in d.keys() if k != 'event_markers']
    rates = {k: d[k]['Fs'] for k in channel_keys}
    unique_rates = set(rates.values())

    if len(unique_rates) > 1:
        details = ', '.join(f"'{k}': {v} Hz" for k, v in rates.items())
        raise ValueError(
            f"All channels must have the same sampling frequency to generate "
            f"a shared timestamp vector. Found: {details}"
        )


def build_timestamp_vector(start_time_local, Fs, n_samples):
    '''Build a unix timestamp vector from a local start time.

    Args:
        start_time_local: timezone-aware datetime object
        Fs: sampling frequency in Hz
        n_samples: total number of samples

    Returns:
        numpy array of unix timestamps (float64 seconds since epoch), one per sample

    MATLAB usage:
        datetime(d.timestamps_local, 'ConvertFrom', 'epochtime', 'TicksPerSecond', 1)
    '''
    start_epoch = start_time_local.timestamp()
    step_seconds = 1.0 / Fs
    return start_epoch + np.arange(n_samples, dtype=np.float64) * step_seconds


def parse_data(data):
    '''Read in ACQ file using njvack's bioread package (https://github.com/uwmadison-chm/bioread)

    Returns:
        tuple: (d, start_time) where d is the data dictionary and start_time is
               the recording start time (datetime UTC) from earliest event marker
    '''
    d = {} # new dictionary to be saved with scipy.io

    # Get file start time from earliest event marker
    start_time = data.earliest_marker_created_at
    if start_time is None:
        raise ValueError(
            "ACQ file has no event markers with timestamps. "
            "Cannot determine recording start time for gap calculation."
        )

    # Load signal renaming mapping
    signal_mapping = load_signal_renaming()

    # Add channel data
    for channel in data.channels:
        # Get cleaned channel name
        cleaned_name = clean(channel.name)

        # Apply signal renaming if mapping exists
        final_name = signal_mapping.get(cleaned_name, cleaned_name)

        d[final_name] = {
            'wave': channel.data,
            'Fs': channel.samples_per_second,
            'unit': channel.units,
        }

    # Add event markers. These are the BIOPAC comments placed during recording.
    event_markers = {}
    event_markers['label'] = []
    event_markers['sample_index'] = []
    event_markers['type_code'] = []
    event_markers['type'] = []
    event_markers['channel_number'] = []
    event_markers['channel'] = []
    event_markers['seconds'] = []
    event_markers['minutes'] = []
    event_markers['date_created_utc'] = []

    valid_events = [i for i in data.event_markers if i.type_code != 'nrto']
    for event in valid_events:

        # Error check for missing timestamp (REQUIRED for CSV export)
        if event.date_created_utc is None:
            raise ValueError(
                f"Event '{event.text}' at sample {event.sample_index} is missing "
                f"date_created_utc timestamp. Cannot export CSV."
            )

        [setattr(event, key, np.nan) for key in event.__dict__.keys() if getattr(event, key) == None]

        event_markers['label'].append(event.text)
        event_markers['sample_index'].append(event.sample_index+1) # +1 since MATLAB is indexed from 1, not 0
        event_markers['type_code'].append(event.type_code)
        event_markers['type'].append(event.type)
        event_markers['channel_number'].append(event.channel_number)
        event_markers['channel'].append(event.channel)

        # Calculate time values (will be recalculated after concatenation if needed)
        Fs = data.channels[0].samples_per_second
        seconds = event.sample_index / Fs
        event_markers['seconds'].append(seconds)
        event_markers['minutes'].append(seconds / 60)
        event_markers['date_created_utc'].append(event.date_created_utc)

    d['event_markers'] = event_markers

    return d, start_time

def export_event_markers_csv(event_markers, output_path):
    '''Export event markers to CSV with time conversion.'''

    EST = pytz.timezone('US/Eastern')

    # Build DataFrame
    df = pd.DataFrame({
        'label': event_markers['label'],
        'sample_index': event_markers['sample_index'],
        'seconds': event_markers['seconds'],
        'minutes': event_markers['minutes'],
        'time (EST)': [dt.astimezone(EST).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                       for dt in event_markers['date_created_utc']]
    })

    df.to_csv(output_path, index=False)

def cat_multiple_files(d_list, start_times):
    '''Concatenate multiple ACQ file data dictionaries with NaN-filled time gaps.

    Args:
        d_list: List of parsed data dictionaries
        start_times: List of datetime objects (UTC) for each file's start time

    Returns:
        Combined data dictionary with NaN gaps between files and recalculated
        event marker timing.
    '''
    d = d_list[0]

    # Get sampling frequency from first channel (assume all channels same Fs)
    channel_keys = [k for k in d.keys() if k != 'event_markers']
    Fs = d[channel_keys[0]]['Fs']

    # Track cumulative sample count for event marker offset
    cumulative_samples = max([len(d[k]['wave']) for k in channel_keys])

    # Calculate end time of first file
    prev_end_time = start_times[0] + timedelta(seconds=cumulative_samples / Fs)

    for i, d_new in enumerate(d_list[1:], start=1):
        next_start_time = start_times[i]

        # Calculate gap duration
        gap = next_start_time - prev_end_time

        if gap.total_seconds() < 0:
            raise ValueError(
                f"File {i+1} starts before file {i} ends. "
                f"File {i} ends at {prev_end_time.isoformat()}, "
                f"File {i+1} starts at {next_start_time.isoformat()}. "
                f"Overlap of {abs(gap.total_seconds()):.2f} seconds detected."
            )

        # Calculate number of NaN samples to insert
        gap_samples = int(gap.total_seconds() * Fs)

        if gap_samples > 0:
            print(f"Gap detected between file {i} and file {i+1}: {gap.total_seconds():.2f} seconds ({gap_samples} samples)")

        # Insert NaN gap and append new data for each channel
        for key in channel_keys:
            if gap_samples > 0:
                nan_gap = np.full(gap_samples, np.nan)
                d[key]['wave'] = np.append(d[key]['wave'], nan_gap)
            d[key]['wave'] = np.append(d[key]['wave'], d_new[key]['wave'])

        # Update cumulative sample count (including gap)
        offset = cumulative_samples + gap_samples
        cumulative_samples = offset + max([len(d_new[k]['wave']) for k in channel_keys])

        # Concatenate event markers with offset
        for key2 in d['event_markers'].keys():
            if key2 == 'sample_index':
                d['event_markers']['sample_index'] = (
                    d['event_markers']['sample_index'] +
                    [idx + offset for idx in d_new['event_markers']['sample_index']]
                )
            else:
                d['event_markers'][key2] = d['event_markers'][key2] + d_new['event_markers'][key2]

        # Update end time for next iteration
        new_file_samples = max([len(d_new[k]['wave']) for k in channel_keys])
        prev_end_time = next_start_time + timedelta(seconds=new_file_samples / Fs)

    # Recalculate seconds and minutes for all event markers based on final sample indices
    for i in range(len(d['event_markers']['sample_index'])):
        sample_idx = d['event_markers']['sample_index'][i]
        seconds = (sample_idx - 1) / Fs  # -1 to convert from MATLAB 1-indexing
        d['event_markers']['seconds'][i] = seconds
        d['event_markers']['minutes'][i] = seconds / 60

    return d


if __name__ == '__main__':

    args = argument_parser(sys.argv[1:])
    data = [bioread.read_file(i) for i in args.file] # read each file specified in command line

    # Parse data and extract start times
    parsed = [parse_data(i) for i in data]
    d_list = [p[0] for p in parsed]
    start_times = [p[1] for p in parsed]

    if len(d_list) >= 2: # concatenate files if there are more than one
        d = cat_multiple_files(d_list, start_times)
    else:
        d = d_list[0]

    # Validate that all channels share the same sampling frequency
    validate_sampling_rates(d)

    # Add metadata for time vector calculation (always, for both single and multi-file)
    channel_keys = [k for k in d.keys() if k != 'event_markers']
    Fs = d[channel_keys[0]]['Fs']
    d['recording_start_utc'] = start_times[0].isoformat()
    EST = pytz.timezone('US/Eastern')
    start_time_local = start_times[0].astimezone(EST)
    d['recording_start_local'] = start_time_local.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    d['Fs'] = Fs

    # Build local timestamp vector (MATLAB datenums) for all samples
    n_samples = len(d[channel_keys[0]]['wave'])
    d['timestamps_local'] = build_timestamp_vector(start_time_local, Fs, n_samples)

    # Export event markers to CSV (before wrapping and saving to MAT)
    csv_output_path = args.outfile.replace('.mat', '_events.csv')
    try:
        export_event_markers_csv(d['event_markers'], csv_output_path)
        print(f"Event markers exported to: {csv_output_path}")
    except ValueError as e:
        print(f"ERROR: Cannot export event markers CSV: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Warning: Could not export CSV: {e}")

    # Remove datetime objects from event_markers (can't be saved to MATLAB)
    if 'date_created_utc' in d['event_markers']:
        del d['event_markers']['date_created_utc']

    d = {'d': d} # wrap into one MATLAB struct rather than multiple variables

    sio.savemat(args.outfile, d, oned_as='column', do_compression=True)
    print(f"MAT file saved to: {args.outfile}")
