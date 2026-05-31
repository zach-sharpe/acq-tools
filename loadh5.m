function d = loadh5(filename)
%LOADH5 A function to load HDF5 (.h5) files created by acq2h5.py.
%
% USAGE: d = loadh5(filename);
%
% MCIRCC, University of Michigan, Ann Arbor
%
% The loadh5() function reads an .h5 file constructed by acq2h5.py and
% returns a MATLAB structure 'd' that mirrors the convenience of loadfile.m
% (which loads acq2mat.py .mat files), but reflecting the flat HDF5 layout.
%
% Unlike the .mat format -- where each channel is a nested struct with
% .wave/.Fs/.unit -- the .h5 format stores each signal as a bare array under
% /signals/<name>, with Fs and units carried as HDF5 attributes. This loader
% reads each signal into d.signals.<name> (the waveform column vector) and
% collects the per-signal Fs/units into d.Fs_per_signal and d.units_per_signal
% structs keyed by the same names. A flat top-level d.<name> = waveform is also
% provided for quick access.
%
% Output structure 'd':
%   d.signals.<name>          column vector of waveform samples (double)
%   d.<name>                  same waveform, also at top level for convenience
%   d.Fs_per_signal.<name>    sampling frequency (Hz) for that signal
%   d.units_per_signal.<name> unit string for that signal
%   d.timestamps_local        unix-second timestamp per sample (column vector)
%   d.Fs                      shared sampling frequency (Hz)
%   d.recording_start_utc     char, ISO UTC start time
%   d.recording_start_local   char, local start time
%   d.event_markers           table of BIOPAC comments (see below)
%
% NOTE on timestamps: d.timestamps_local holds unix seconds. Convert to
% MATLAB datetime with:
%   datetime(d.timestamps_local, 'ConvertFrom', 'epochtime', 'TicksPerSecond', 1)
%
% EXAMPLES
%
% 1. Plot arterial pressure waveform saved under the signal name "abp"
%
%   d = loadh5(filename);
%   t = (1:numel(d.abp))/d.Fs_per_signal.abp;
%   plot(t, d.abp)
%   xlabel('Time (seconds)')
%   ylabel(sprintf('ABP (%s)', d.units_per_signal.abp))
%
% 2. Annotate with event markers
%
%   text(d.event_markers.sample_index/d.Fs, ...
%       zeros(height(d.event_markers), 1), ...
%       d.event_markers.label, ...
%       'Rotation', 90)
%

d = struct();
d.signals = struct();
d.Fs_per_signal = struct();
d.units_per_signal = struct();

info = h5info(filename, '/signals');

for i = 1:numel(info.Datasets)
    name = info.Datasets(i).Name;
    dpath = ['/signals/' name];

    wave = h5read(filename, dpath);
    wave = double(wave(:));   % force column vector, double precision

    if strcmp(name, 'timestamps_local')
        % Timestamps are not a physiologic signal -- promote to top level.
        d.timestamps_local = wave;
        continue
    end

    d.signals.(name) = wave;
    d.(name) = wave;          % flat convenience accessor

    % Read Fs / units attributes if present on the dataset.
    attrNames = {info.Datasets(i).Attributes.Name};

    if any(strcmp(attrNames, 'Fs'))
        d.Fs_per_signal.(name) = double(h5readatt(filename, dpath, 'Fs'));
    end

    if any(strcmp(attrNames, 'units'))
        units = h5readatt(filename, dpath, 'units');
        if iscell(units), units = units{1}; end
        d.units_per_signal.(name) = char(units);
    end
end

% Top-level scalar datasets.
d.Fs = double(h5read(filename, '/Fs'));

start_utc = h5read(filename, '/recording_start_utc');
if iscell(start_utc), start_utc = start_utc{1}; end
d.recording_start_utc = char(start_utc);

start_local = h5read(filename, '/recording_start_local');
if iscell(start_local), start_local = start_local{1}; end
d.recording_start_local = char(start_local);

% event_markers is a JSON-encoded scalar string -> decode to a table.
em_json = h5read(filename, '/event_markers');
if iscell(em_json), em_json = em_json{1}; end
em = jsondecode(char(em_json));

% jsondecode yields a struct whose fields are column-cell/array vectors.
% struct2table turns those parallel arrays into a table directly.
d.event_markers = struct2table(em);

end
