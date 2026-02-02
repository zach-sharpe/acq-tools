#!/usr/bin/env python
'''
acq2mat_gui.py

GUI wrapper for acq2mat.py using FreeSimpleGUI.
Allows easy file selection and conversion of ACQ files to MATLAB format.
'''

import os
import sys
import FreeSimpleGUI as sg
import subprocess
from pathlib import Path


def create_layout():
    '''Create the GUI layout.'''

    sg.theme('DefaultNoMoreNagging')

    layout = [
        [sg.Text('ACQ to MAT Converter', font=('Helvetica', 16, 'bold'))],
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
        [sg.Input(key='-OUTFILE-', size=(50, 1)), sg.Text('.mat')],
        [sg.Text('Leave blank to auto-generate from folder name',
                 font=('Helvetica', 9, 'italic'), text_color='gray')],

        [sg.Text('')],  # Spacer
        [sg.HorizontalSeparator()],

        # Action buttons
        [sg.Button('Convert', size=(10, 1), button_color=('white', 'green')),
         sg.Button('Cancel', size=(10, 1))],

        [sg.Text('')],  # Spacer

        # Status/output area
        [sg.Multiline(key='-OUTPUT-', size=(70, 10), disabled=True,
                     autoscroll=True, background_color='white', text_color='black')]
    ]

    return layout


def generate_default_filename(folder_path):
    '''
    Generate default .mat filename from folder name.
    Example: "/path/to/2025-10-11" -> "2025-10-11.mat"
    '''
    if not folder_path:
        return ""

    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    return f"{folder_name}.mat"


def run_conversion(acq_files, output_folder, output_filename, window):
    '''
    Execute acq2mat.py conversion.

    Args:
        acq_files (str): Semicolon-separated list of ACQ file paths
        output_folder (str): Output directory path
        output_filename (str): Output filename (with or without .mat)
        window: FreeSimpleGUI window object for updating output
    '''
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
        output_filename = generate_default_filename(output_folder)
        window['-OUTPUT-'].update(f"Using auto-generated filename: {output_filename}\n", append=True)

    # Ensure .mat extension
    if not output_filename.endswith('.mat'):
        output_filename += '.mat'

    # Build full output path
    output_path = os.path.join(output_folder, output_filename)

    # Build command
    script_dir = os.path.dirname(os.path.abspath(__file__))
    acq2mat_script = os.path.join(script_dir, 'acq2mat.py')

    cmd = [sys.executable, acq2mat_script] + file_list + ['-o', output_path]

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

    window = sg.Window('ACQ2MAT Converter', create_layout(), finalize=True)

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, 'Cancel'):
            break

        # Auto-update filename when folder changes
        if event == '-OUTFOLDER-':
            if values['-OUTFOLDER-'] and not values['-OUTFILE-']:
                default_name = generate_default_filename(values['-OUTFOLDER-'])
                # Don't auto-fill, just show it would be used
                window['-OUTPUT-'].update(
                    f"Folder selected: {values['-OUTFOLDER-']}\n"
                    f"Default filename will be: {default_name}\n\n",
                    append=True)

        # Convert button
        if event == 'Convert':
            window['-OUTPUT-'].update('')  # Clear output
            run_conversion(
                values['-FILES-'],
                values['-OUTFOLDER-'],
                values['-OUTFILE-'],
                window
            )

    window.close()


if __name__ == '__main__':
    main()
