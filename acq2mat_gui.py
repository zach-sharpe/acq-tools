#!/usr/bin/env python
'''
acq2mat_gui.py

GUI wrapper around convert_acq.py using FreeSimpleGUI.
Allows easy file selection and conversion of ACQ files to either MATLAB
(.mat, "Classic") or HDF5 (.h5, "New") format. The two formats share the
same file/folder selection; the chosen Convert button decides the output
format. The actual conversion is performed in-process by convert_acq(), so
the GUI and the CLI scripts all run the exact same pipeline.
'''

import FreeSimpleGUI as sg

from convert_acq import convert_acq, default_filename, FORMATS as _CORE_FORMATS

# GUI display labels per format, keyed to convert_acq's canonical FORMATS so the
# extension (.mat/.h5) has a single source of truth.
_LABELS = {'mat': '.mat (Classic)', 'h5': '.h5 (New)'}
FORMATS = {
    fmt: {'ext': _CORE_FORMATS[fmt]['ext'], 'label': _LABELS[fmt]}
    for fmt in _CORE_FORMATS
}


def create_layout():
    '''Create the GUI layout.'''

    sg.theme('DefaultNoMoreNagging')

    layout = [
        [sg.Text('ACQ Converter', font=('Helvetica', 16, 'bold'))],
        [sg.HorizontalSeparator()],

        # File selection
        [sg.Text('Select ACQ files to concatenate:', size=(25, 1))],
        [sg.Input(key='-FILES-', disabled=True, size=(50, 1)),
         sg.FilesBrowse(button_text='Browse...', file_types=(("ACQ Files", "*.acq"),))],

        [sg.Text('')],  # Spacer

        # Output folder selection
        [sg.Text('Select output folder:', size=(25, 1))],
        [sg.Input(key='-OUTFOLDER-', size=(50, 1), enable_events=True),
         sg.FolderBrowse(button_text='Browse...')],

        [sg.Text('')],  # Spacer

        # Output filename
        [sg.Text('Output filename (optional):', size=(25, 1))],
        [sg.Input(key='-OUTFILE-', size=(50, 1)),
         sg.Text('+ format extension', font=('Helvetica', 9, 'italic'), text_color='gray')],
        [sg.Text('Leave blank to auto-generate from folder name. The extension '
                 'is set by the Convert button you press.',
                 font=('Helvetica', 9, 'italic'), text_color='gray')],

        [sg.Text('')],  # Spacer
        [sg.HorizontalSeparator()],

        # Action buttons -- one per output format, plus Cancel
        [sg.Button('Convert to .mat', key='-CONVERT-MAT-', size=(14, 1),
                   button_color=('white', 'green')),
         sg.Button('Convert to .h5', key='-CONVERT-H5-', size=(14, 1),
                   button_color=('white', '#1f6aa5')),
         sg.Button('Cancel', size=(10, 1))],

        [sg.Text('')],  # Spacer

        # Status/output area
        [sg.Multiline(key='-OUTPUT-', size=(70, 10), disabled=True,
                     autoscroll=True, background_color='white', text_color='black')]
    ]

    return layout


def generate_default_filename(folder_path, ext):
    '''
    Generate default output filename from folder name.
    Example: "/path/to/2025-10-11", ".mat" -> "2025-10-11.mat"

    Thin wrapper over convert_acq.default_filename so the GUI and the converter
    share one auto-naming rule.
    '''
    return default_filename(folder_path, ext)


def run_conversion(acq_files, output_folder, output_filename, fmt, window):
    '''
    Convert the selected ACQ files for the requested output format.

    Runs the conversion in-process via convert_acq() -- the same pipeline the
    CLI scripts use -- and reports progress into the GUI output pane.

    Args:
        acq_files (str): Semicolon-separated list of ACQ file paths (as returned
            by FreeSimpleGUI's FilesBrowse).
        output_folder (str): Output directory path.
        output_filename (str): Output filename (with or without the extension).
        fmt (str): Output format key -- 'mat' or 'h5' (see FORMATS).
        window: FreeSimpleGUI window object for updating output.

    Returns:
        bool: True on success, False on any validation or conversion error.
    '''
    # Validate inputs (the GUI shows these as friendly messages rather than
    # letting convert_acq raise).
    if not acq_files:
        window['-OUTPUT-'].update("ERROR: No ACQ files selected.\n", append=True)
        return False

    if not output_folder:
        window['-OUTPUT-'].update("ERROR: No output folder selected.\n", append=True)
        return False

    # FreeSimpleGUI returns a semicolon-separated string; convert_acq wants a list.
    file_list = [f.strip() for f in acq_files.split(';') if f.strip()]

    window['-OUTPUT-'].update("Running conversion...\n\n", append=True)

    try:
        result = convert_acq(file_list, output_folder, output_filename, fmt)
    except Exception as e:
        # convert_acq raises ValueError / FileNotFoundError on bad input and
        # hard pipeline errors (e.g. sampling-rate mismatch).
        window['-OUTPUT-'].update(f"\nERROR: {e}\n", append=True)
        return False

    # Forward the converter's progress lines, then a success summary.
    for msg in result.messages:
        window['-OUTPUT-'].update(msg + "\n", append=True)

    window['-OUTPUT-'].update(f"\nSUCCESS! Output saved to:\n{result.output_path}\n", append=True)
    if result.csv_path:
        window['-OUTPUT-'].update(f"CSV comments file saved to:\n{result.csv_path}\n", append=True)
    return True


def main():
    '''Main GUI event loop.'''

    window = sg.Window('ACQ Converter', create_layout(), finalize=True)

    # Map Convert button events to their output format key.
    convert_events = {'-CONVERT-MAT-': 'mat', '-CONVERT-H5-': 'h5'}

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, 'Cancel'):
            break

        # Auto-update filename when folder changes
        if event == '-OUTFOLDER-':
            if values['-OUTFOLDER-'] and not values['-OUTFILE-']:
                # Show the base name; the extension depends on the Convert
                # button pressed, so display it without one.
                base_name = generate_default_filename(values['-OUTFOLDER-'], '')
                window['-OUTPUT-'].update(
                    f"Folder selected: {values['-OUTFOLDER-']}\n"
                    f"Default filename will be: {base_name}.mat / {base_name}.h5\n\n",
                    append=True)

        # Convert buttons (one per output format)
        if event in convert_events:
            window['-OUTPUT-'].update('')  # Clear output
            run_conversion(
                values['-FILES-'],
                values['-OUTFOLDER-'],
                values['-OUTFILE-'],
                convert_events[event],
                window
            )

    window.close()


if __name__ == '__main__':
    main()
