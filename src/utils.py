
import pandas as pd
import pathlib

def read_txt_file(path):
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
    :param dataframe:
    :param filepath:
    :return:
    """
    try:
        excel_writer = pd.ExcelWriter(pathlib.Path(filepath), engine="xlsxwriter")
        dataframe.to_excel(excel_writer, sheet_name="Main", index=False)
        excel_writer.save()
    except Exception as excep:
        print(f"Error writing excel file from {filepath}: {excep}")