import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build    
import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np
import os
import json
import time
# use creds to create a client to interact with the Google Drive API
scope = ['https://www.googleapis.com/auth/analytics.readonly',
      'https://www.googleapis.com/auth/drive',
      'https://www.googleapis.com/auth/spreadsheets',
      ]#['https://spreadsheets.google.com/feeds']
#creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
creds = ServiceAccountCredentials.from_json_keyfile_name('./creds/online_notebook_drive_api.json', scope)
client = gspread.authorize(creds)
creds_drive = ServiceAccountCredentials.from_json_keyfile_name('./creds/online_notebook_drive_api.json', scope)

#%% open 

def fetch_lastmodify_time(spreadsheetname):
    modifiedtime = None
    ID = None
    service = build('drive', 'v3', credentials=creds)
    wb = client.open(spreadsheetname)
    ID = wb.id
    if ID:
        modifiedtime = service.files().get(fileId = ID,fields = 'modifiedTime').execute()
    return modifiedtime

def fetch_sheet_titles(spreadsheetname):
    wb = client.open(spreadsheetname)
    sheetnames = list()
    worksheets = wb.worksheets()
    for sheet in worksheets:
        sheetnames.append(sheet.title)
    return sheetnames
#%
def fetch_sheet(spreadsheet_name,sheet_title,transposed = False):
    #%
    wb = client.open(spreadsheet_name)
# =============================================================================
#     sheetmetadata = wb.fetch_sheet_metadata()
#     sheetnames = list()
#     #worksheets = wb.worksheets()
#     for sheet in sheetmetadata['sheets']:
#         sheetnames.append(sheet['properties']['title'])
#         #%
#     if sheet_title in sheetnames:
# =============================================================================
    try:
# =============================================================================
#         idx_now = sheetnames.index(sheet_title)
#         if idx_now > -1:
# =============================================================================
        #%%
        if transposed:
            params = {'majorDimension':'COLUMNS'}
        else:
            params = {'majorDimension':'ROWS'}
        temp = wb.values_get(sheet_title+'!A1:QQ500',params)
        temp = temp['values']
        header = temp.pop(0)
        data = list()
        for row in temp:
            data.append(row)
        try:
            df = pd.DataFrame(data, columns = header)
        except:
            print([spreadsheet_name,sheet_title])
            print(header)
            print(data)
            #%%
        return df
# =============================================================================
#         else:
#             return None
# =============================================================================
    except:
        return None
    
def fetch_lab_metadata(ID):
    #%
    wb = client.open("Lab metadata")
    sheetnames = list()
    while True:
        try:
            worksheets = wb.worksheets()
            break
        except:
            print('quota exceeded?, waiting 15s')
            time.sleep(15)
    for sheet in worksheets:
        sheetnames.append(sheet.title)
        #%
    if ID in sheetnames:
        idx_now = sheetnames.index(ID)
        if idx_now > -1:
            params = {'majorDimension':'ROWS'}
            temp = wb.values_get(ID+'!A1:QQ500',params)
            temp = temp['values']
            header = temp.pop(0)
            data = list()
            for row in temp:
                if len(row) < len(header):
                    row.append('')
                if len(row) == len(header):
                    data.append(row)
            df = pd.DataFrame(data, columns = header)
            return df
        else:
            return None
    else:
        return None

def update_metadata(notebook_name,metadata_dir,transposed = False):
    #%%
    lastmodify = fetch_lastmodify_time(notebook_name)
    try:
        with open(os.path.join(metadata_dir,'last_modify_time.json')) as timedata:
            lastmodify_prev = json.loads(timedata.read())
    except:
        lastmodify_prev = None
    if lastmodify != lastmodify_prev:
        print('updating metadata from google drive')
        sessions = fetch_sheet_titles(notebook_name)
        for session in sessions:
            while True:
                try:    
                    df_wr = fetch_sheet(notebook_name,session,transposed)
                    break
                except gspread.exceptions.APIError as err:
                    print(err)
                    print('quota exceeded at table {}, waiting 150s'.format(session))
                    time.sleep(150)
            if type(df_wr) == pd.DataFrame:
                df_wr.to_csv(os.path.join(metadata_dir,'{}.csv'.format(session))) 
                #%
        with open(os.path.join(metadata_dir,'last_modify_time.json'), "w") as write_file:
            json.dump(lastmodify, write_file)
            #%
        print('metadata updated')
    else:
        print('metadata is already up to date')