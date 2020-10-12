import pandas as pd
import sys


SourceFilePath = sys.argv[1]
SourceFileName = sys.argv[2]
DestinationPath = 'C:\\Users\\HP India\\Documents\\Python Programs\\ConvertExcelToPsv'

SourceFile = SourceFilePath+'\\'+SourceFileName
xls = pd.ExcelFile(SourceFile, engine='pyxlsb')

sheet_names = xls.sheet_names

for sheet_name in sheet_names:
    DataFrame = pd.read_excel(SourceFile, sheet_name, engine='pyxlsb')
    DataFrame = DataFrame.replace('\"', '', regex=True)
    DataFrame = DataFrame.replace('\|', '', regex=True)
    DestinationFile = DestinationPath+'//'+SourceFileName[:SourceFileName.index(".xlsb")]+'_'+sheet_name+'.psv'
    DataFrame.to_csv(DestinationFile, sep='|', index=False)
