def sort_strings_by_length(input_file, output_file):
    """
    Reads strings from a file, sorts them by length, and writes the sorted strings to another file.

    :param input_file: Path to the input file containing strings (one per line).
    :param output_file: Path to the output file to write the sorted strings.
    """
    try:
        # Read lines from the input file
        with open(input_file, 'r') as infile:
            strings = infile.readlines()

        # Remove whitespace characters (like newlines) from the end of each line
        strings = [s.strip().replace("[","").replace("]","").replace(",","") for s in strings]

        strings = [ ''.join(['z' + char if char.isdigit() else char for char in s]) for s in strings]
        # Sort strings by length
        strings.sort()
        strings.sort(key=len)

        # Write the sorted strings to the output file
        with open(output_file, 'w') as outfile:
            outfile.write('\n'.join(strings))

        print(f"Strings sorted by length have been written to {output_file}.")

    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Python script to concatenate every six lines into one line within a file

def concatenate_six_lines(input_file, output_file):
    """
    Reads a file and concatenates every six lines into one line, writing the results to another file.

    :param input_file: Path to the input file containing lines of text.
    :param output_file: Path to the output file to write the concatenated lines.
    """
    try:
        # Open the input file for reading
        with open(input_file, 'r') as infile:
            lines = infile.readlines()

        # Open the output file for writing
        with open(output_file, 'w') as outfile:
            for i in range(0, len(lines), 5):
                # Concatenate six lines, stripping whitespace and joining with a space
                concatenated_line = ' '.join(line.strip() for line in lines[i:i+5])
                outfile.write(concatenated_line + '\n')

        print(f"Lines concatenated successfully and written to {output_file}.")

    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    ifs = [
        "train_com.csv",
        "train_ide.csv",
        "test_com.csv",
        "test_ide.csv",
    ]
    ofs = [
        "out_train_com.csv",
        "out_train_ide.csv",
        "out_test_com.csv",
        "out_test_ide.csv",
    ]
    for input_file, output_file in zip(ifs,ofs):
        sort_strings_by_length(input_file, output_file)
        concatenate_six_lines(output_file,output_file)
