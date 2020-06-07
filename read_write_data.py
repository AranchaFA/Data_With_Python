import numpy as np
import pandas as pd

# READ FROM CSV
print('-- READING CSV --')
# If we omit delimiter parameter, by default ','
print(pd.read_csv('data.csv', header=0, nrows=2, delimiter=';')) # Only 2 first rows
print(pd.read_csv('data.csv', header=0, delimiter=';'))
print(pd.read_csv('data.csv', header=None, delimiter=';')) # Header = 0,1,2,3,...
print(pd.read_csv('data.csv', delimiter=';')) # No header
data_frame_csv1 = pd.DataFrame(pd.read_csv('data.csv', header=0, delimiter=';'))
data_frame_csv2 = pd.read_csv('data.csv', delimiter=';')
print(data_frame_csv1[-2:][1:2]) # (df[-2:])[1:2]
print(data_frame_csv1.loc(0))
print(data_frame_csv1.loc(0)[0])
print(data_frame_csv1.loc[0])
print(data_frame_csv1['COL2'].loc[0])

# WRITE TO A CSV
print('\n-- WRITTING CSV --\n... :)')
data_frame_csv1.to_csv('written.csv')
# Header is always written!
data_frame_csv2.to_csv('written2.csv')
data_frame_csv2[1:].to_csv('written3.csv') # It trims first row, not header!

# READ FROM EXCEL
print('\n-- READING EXCEL --')
# Dependecies -> pip install xlrd or conda install -c anaconda xlrd (inside project folder, with activated environment)
file_xlsx = pd.ExcelFile('data.xlsx')
print(pd.read_excel('data.xlsx', sheet_name='Sheet1')) # Not shapely
print(pd.read_excel(file_xlsx, sheet_name='Sheet2')) # Shapely
data_frame_excel1 = pd.DataFrame(pd.read_excel('data.xlsx', sheet_name='Sheet2'))


# WRITE TO AN EXCEL