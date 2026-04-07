import pandas as pd
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

jsonFilePath = "C:/Dir/IITM-M.Tech/Wireless Communication/MiniProject/zone_cell_data/"
# jsonFileNameWithoutExtension = "zone1"
jsonFileNameWithoutExtension = "zone2"
jsonFileNameWithPath = jsonFilePath+jsonFileNameWithoutExtension+".json"

with open(jsonFileNameWithPath) as f:
    dataZone1 = json.load(f)

rows = []
# print(type(dataZone1))
# print(len(dataZone1))
# print(len(dataZone2))

#print(dataZone1[0]['GPS_data']['Latitude'])
#print(dataZone1[0]['registered']['Provider'])
# print(dataZone1[0]['neighbors'][0]['rsrp'])   #This is list of neighbors


# zoneOneData = pd.DataFrame(columns = ['Timestamp', 'Lat', 'Long', 'Type','Provider' ,'ci', 'pci', 'tac', 
#                                       'earfcn', 'mcc', 'mnc', 'rsrp', 'rsrq', 'rssnr', 'rssi', 'level'])


zoneOneDataList = []   #Create empty List

for i in dataZone1:
    timestamp = i['Timestamp']
    lat = i["GPS_data"]["Latitude"]
    lon = i["GPS_data"]["Longitude"]
    registered = i['registered']
    #print(type(registered))
    localList = [timestamp, lat, lon, 'registered',registered.get('Provider'), 
                                        registered.get('ci'),
                                        registered.get('pci'),
                                        registered.get('tac'),
                                        registered.get('earfcn'),
                                        registered.get('mcc'),
                                        registered.get('mnc'),
                                        registered.get('rsrp'),
                                        registered.get('rsrq'),
                                        registered.get('rssnr'),
                                        registered.get('rssi'),
                                        registered.get('level')]
    zoneOneDataList.append(localList)
    for j in i['neighbors']:
        localList = []
        localList = [timestamp, lat, lon, 'neighbour',j.get('Provider'),
                                            j.get('ci'),
                                            j.get('pci'),
                                            j.get('tac'),
                                            j.get('earfcn'),
                                            j.get('mcc'),
                                            j.get('mnc'),
                                            j.get('rsrp'),
                                            j.get('rsrq'),
                                            j.get('rssnr'),
                                            j.get('rssi'),
                                            j.get('level')]
        zoneOneDataList.append(localList)


# Convert to DataFrame
zoneOneData = pd.DataFrame(zoneOneDataList, columns=['Timestamp', 'Lat', 'Long', 'Type','Provider','ci','pci','tac',
                                             'earfcn','mcc','mnc','rsrp','rsrq','rssnr','rssi','level'])

zoneOneData.head()
zoneOneData.to_csv(jsonFilePath+jsonFileNameWithoutExtension+".csv", index=False)
print("Zone Json Converted to CSV");
