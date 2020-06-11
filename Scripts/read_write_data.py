import numpy as np
import pandas as pd
import openpyxl
import xlsxwriter

# READ FROM CSV
print('-- READING CSV --')
# If we omit delimiter parameter, by default ','
print(pd.read_csv('../files/data.csv', header=0, nrows=2, delimiter=';')) # Only 2 first rows
print(pd.read_csv('../files/data.csv', header=0, delimiter=';'))
print(pd.read_csv('../files/data.csv', header=None, delimiter=';')) # Header = 0,1,2,3,...
print(pd.read_csv('../files/data.csv', delimiter=';')) # No header
data_frame_csv1 = pd.DataFrame(pd.read_csv('../files/data.csv', header=0, delimiter=';'))
data_frame_csv2 = pd.read_csv('../files/data.csv', delimiter=';')
print(data_frame_csv1[-2:][1:2]) # (data_frame[-2:])[1:2]
print(data_frame_csv1.loc(0))
print(data_frame_csv1.loc(0)[0])
print(data_frame_csv1.loc[0])
print(data_frame_csv1['COL2'].loc[0])

# WRITE TO CSV
print('\n-- WRITTING CSV --\n... :)')
data_frame_csv1.to_csv('../files/written.csv')
# Header is always written!
data_frame_csv2.to_csv('../files/written2.csv')
data_frame_csv2[1:].to_csv('../files/written3.csv') # It trims first row, not header!

# READ FROM EXCEL
print('\n-- READING EXCEL --')
# Dependecies -> pip install xlrd or conda install -c anaconda xlrd (inside project folder, with activated environment)
file_xlsx = pd.ExcelFile('../files/data.xlsx')
print(file_xlsx.sheet_names)
print(pd.read_excel('../files/data.xlsx', sheet_name='Sheet1')) # Not shapely
print(pd.read_excel(file_xlsx, sheet_name='Sheet2')) # Shapely
data_frame_excel1 = pd.DataFrame(pd.read_excel(file_xlsx, sheet_name='Sheet2'))

# WRITE TO EXCEL
data_frame_excel2 = pd.DataFrame({'C1':['*1','*2'],
                                  'C2':['+1','+2'],
                                  'C3':['-1','-2']})
print(data_frame_excel2)

# Creating new file / Overwritting existing one
data_frame_excel2.to_excel('../files/newFile.xlsx', sheet_name='Sheet3', index=False)

# Adding to existing file (or creating new one if file_name doesn't exist)
def write_df_to_excel(file_name, data_frame, sheet_name='Sheet1', start_row=None,
                      overwrite_sheet=False,
                      **to_excel_kwargs):
    """
    Append data from a DataFrame [df] to existing Excel file [filename]
    into [sheet_name] Sheet.
    If [sheet_name] doesn't exist in [filename], then this function will create it.
    If [filename] doesn't exist, then this function will create it.

    Parameters:
      file_name : File path or existing ExcelWriter
                 (Example: '/path/to/file.xlsx')
      data_frame : dataframe to save to workbook
      sheet_name : Name of sheet in which will write DataFrame data.
                   (default: 'Sheet1')
      start_row : upper left cell row to dump data frame.
                 Per default (startrow=None) append after last row in existing
                 sheets, and at first row (startrow=0) in new sheets
      overwrite_sheet : overwrite [sheet_name] with DataFrame-to-Excel file
      to_excel_kwargs : arguments which will be passed to `DataFrame.to_excel()`
                        [can be dictionary]

    Returns: None
    """
    from openpyxl import load_workbook

    # ignore [engine] parameter if it was passed
    if 'engine' in to_excel_kwargs:
        to_excel_kwargs.pop('engine')
    writer = pd.ExcelWriter(file_name, engine='openpyxl')

    try:
        # open existing workbook
        writer.book = load_workbook(file_name)

        # start_row -> by default get last row if sheet actually exists
        if start_row is None and sheet_name in writer.book.sheetnames and not overwrite_sheet:
            start_row = writer.book[sheet_name].max_row

        # truncate sheet -> remove existing and create an empty one with same name and index
        if overwrite_sheet and sheet_name in writer.book.sheetnames:
            idx = writer.book.sheetnames.index(sheet_name)
            writer.book.remove(writer.book.worksheets[idx])
            writer.book.create_sheet(sheet_name, idx)

        # copy existing sheets -> {title:sheet}
        writer.sheets = {ws.title:ws for ws in writer.book.worksheets}
    except FileNotFoundError:
        pass # file will be automatically created

    if start_row is None:
        start_row = 0

    # indexes and column header -> by default only headers when truncating
    if 'index' not in to_excel_kwargs:
        to_excel_kwargs['index'] = False
    if 'header' not in to_excel_kwargs:
        to_excel_kwargs['header'] = overwrite_sheet

    # write the new data
    data_frame.to_excel(writer, sheet_name, startrow=start_row, **to_excel_kwargs)

    # save the workbook
    writer.save()

# TESTING
# New file
write_df_to_excel('../files/datatest.xlsx', data_frame_excel2, sheet_name='MyTestSheet', header=True)
# New sheet in existing file
write_df_to_excel('../files/datatest.xlsx', data_frame_excel2, sheet_name='MyTestSheet2', header=True) 
