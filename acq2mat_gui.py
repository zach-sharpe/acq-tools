#!/usr/bin/env python
'''
acq2mat_gui.py

GUI wrapper for acq2mat.py and acq2h5.py using FreeSimpleGUI.
Allows easy file selection and conversion of ACQ files to either MATLAB
(.mat, "Classic") or HDF5 (.h5, "New") format. The two formats share the
same file/folder selection; the chosen Convert button decides the output
format and which converter script is invoked.
'''

import os
import sys
import FreeSimpleGUI as sg
import subprocess
from pathlib import Path


# Map output format -> (file extension, converter script). Both converters
# share the same CLI: <acq files...> -o <output path>.
FORMATS = {
    'mat': {'ext': '.mat', 'script': 'acq2mat.py', 'label': '.mat (Classic)'},
    'h5':  {'ext': '.h5',  'script': 'acq2h5.py',  'label': '.h5 (New)'},
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
    '''
    if not folder_path:
        return ""

    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    return f"{folder_name}{ext}"


def run_conversion(acq_files, output_folder, output_filename, fmt, window):
    '''
    Execute the converter for the requested output format.

    Args:
        acq_files (str): Semicolon-separated list of ACQ file paths
        output_folder (str): Output directory path
        output_filename (str): Output filename (with or without the extension)
        fmt (str): Output format key -- 'mat' or 'h5' (see FORMATS).
        window: FreeSimpleGUI window object for updating output
    '''
    ext = FORMATS[fmt]['ext']
    script_name = FORMATS[fmt]['script']

    # Validate inputs
    if not acq_files:
        window['-OUTPUT-'].update("ERROR: No ACQ files selected.\n", append=True)
        return False

    if not output_folder:
        window['-OUTPUT-'].update("ERROR: No output folder selected.\n", append=True)
        return False

    # Parse file list (FreeSimpleGUI returns semicolon-separated string)
    file_list = [f.strip() for f in acq_files.split(';') if f.strip()]

    # Determine output filename
    if not output_filename:
        output_filename = generate_default_filename(output_folder, ext)
        window['-OUTPUT-'].update(f"Using auto-generated filename: {output_filename}\n", append=True)

    # Ensure the correct extension for the chosen format
    if not output_filename.endswith(ext):
        output_filename += ext

    # Build full output path
    output_path = os.path.join(output_folder, output_filename)

    # Build command
    script_dir = os.path.dirname(os.path.abspath(__file__))
    converter_script = os.path.join(script_dir, script_name)

    cmd = [sys.executable, converter_script] + file_list + ['-o', output_path]

    window['-OUTPUT-'].update(f"Running conversion...\n", append=True)
    window['-OUTPUT-'].update(f"Command: {' '.join(cmd)}\n\n", append=True)

    # Execute
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Display stdout
        if result.stdout:
            window['-OUTPUT-'].update(result.stdout + "\n", append=True)

        # Display stderr
        if result.stderr:
            window['-OUTPUT-'].update("STDERR:\n" + result.stderr + "\n", append=True)

        # Check return code
        if result.returncode == 0:
            window['-OUTPUT-'].update(f"\nSUCCESS! Output saved to:\n{output_path}\n", append=True)
            window['-OUTPUT-'].update(f"CSV comments file also saved to output folder.\n", append=True)
            return True
        else:
            window['-OUTPUT-'].update(f"\nERROR: Conversion failed with return code {result.returncode}\n", append=True)
            return False

    except Exception as e:
        window['-OUTPUT-'].update(f"\nEXCEPTION: {str(e)}\n", append=True)
        return False


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
