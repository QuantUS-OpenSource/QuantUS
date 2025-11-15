#!/bin/bash

# Navigate to the target directory
cd ./src || exit

# Find all .ui files in subdirectories and convert them to .py files
find . -name "*.ui" | while read -r ui_file; do
    py_file="${ui_file%.ui}_ui.py"
    echo "Converting $ui_file to $py_file"
    pyuic6 "$ui_file" -o "$py_file"
done