"""
Function to define helpers for the training
"""
import pathlib
import pandas as pd
import numpy as np
import trax


def append_eos(stream):
    """
    Generator helper function to append eos to each sentence

    Args:
        stream:
    Returns:
    """
    #Append eos at the end of each sentence.
    eos = 1
    for (inputs, targets) in stream:
        inputs_with_eos = list(inputs) + [eos]
        targets_with_eos = list(targets) + [eos]
        yield np.array(inputs_with_eos), np.array(targets_with_eos)

def tokenize(input_str, vocab_file=None, vocab_dir=None):
    """
    Encodes a string to an array of integers

    Args:
        input_str (str): human-readable string to encode
        vocab_file (str): filename of the vocabulary text file
        vocab_dir (str): path to the vocabulary file
    Returns:
        numpy.ndarray: tokenized version of the input string
    """
    # Set the encoding of the "end of sentence" as 1
    eos = 1
    # Use the trax.data.tokenize method. It takes streams and returns streams,
    # we get around it by making a 1-element stream with `iter`.
    inputs = next(trax.data.tokenize(iter([input_str]),
                                     vocab_file=vocab_file, vocab_dir=vocab_dir))
    # Mark the end of the sentence with eos
    inputs = list(inputs) + [eos]
    # Adding the batch dimension to the front of the shape
    batch_inputs = np.reshape(np.array(inputs), [1, -1])
    return batch_inputs


def detokenize(integers, vocab_file=None, vocab_dir=None):
    """
    Decodes an array of integers to a human readable string

    Args:
        integers (numpy.ndarray): array of integers to decode
    	vocab_file (str): filename of the vocabulary text file
    	vocab_dir (str): path to the vocabulary file
    Returns:
        str: the decoded sentence.
    """

    # Remove the dimensions of size 1
    integers = list(np.squeeze(integers))
    # Set the encoding of the "end of sentence" as 1
    eos = 1
    # Remove the eos to decode only the original tokens
    if eos in integers:
        integers = integers[:integers.index(eos)]
    return trax.data.detokenize(integers, vocab_file=vocab_file, vocab_dir=vocab_dir)


def read_txt_file(path):
    """
    Function to read text file

    Args:
    	path: path to the file
    Returns:
        string content file
    """

    text = ""
    with open(path) as file:
        line = file.readline()
        while line:
            text = text + line
            line = file.readline()
    return text


def write_excel(dataframe, filepath):
    """
    Write pandas dataframe to excel

    Args:
    	dataframe: dataframe pandas
    	filepath: file to write
    Returns:
    """
    try:
        excel_writer = pd.ExcelWriter(pathlib.Path(filepath), engine="xlsxwriter")
        dataframe.to_excel(excel_writer, sheet_name="Main", index=False)
        excel_writer.save()
    except Exception as excep:
        print(f"Error writing excel file from {filepath}: {excep}")
