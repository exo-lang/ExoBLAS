#!/usr/bin/env python3

import os

def count_lines_in_file(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        in_multiline_comment = False
        code_lines = 0

        for line in file:
            stripped_line = line.strip()

            if in_multiline_comment:
                if '*/' in stripped_line:
                    # End of multi-line comment
                    in_multiline_comment = False
                    stripped_line = stripped_line.split('*/', 1)[1]
                else:
                    continue  # Skip everything inside multi-line comments

            # Remove single-line comments
            if '//' in stripped_line:
                stripped_line = stripped_line.split('//', 1)[0]

            # Check for the start of a multi-line comment
            if '/*' in stripped_line:
                if '*/' in stripped_line:
                    # Multi-line comment starts and ends on the same line
                    before_comment = stripped_line.split('/*', 1)[0]
                    after_comment = stripped_line.split('*/', 1)[1]
                    stripped_line = before_comment + after_comment
                else:
                    # Multi-line comment starts here and continues
                    stripped_line = stripped_line.split('/*', 1)[0]
                    in_multiline_comment = True

            # Count the line if it's not empty after stripping comments and whitespace
            if stripped_line.strip():
                code_lines += 1

    return code_lines

def main():
    total_lines = 0

    for root, _, files in os.walk('.'):
        for filename in files:
            if filename.endswith('.c'):
                filepath = os.path.join(root, filename)
                lines = count_lines_in_file(filepath)
                print(f"{filepath}: {lines} lines")
                total_lines += lines

    print(f"Total lines of code: {total_lines}")

if __name__ == "__main__":
    main()

