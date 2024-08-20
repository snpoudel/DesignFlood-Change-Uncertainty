import requests
import pandas as pd

#read list of stations for MA
stn = pd.read_csv('data/MA_stations.csv')
#loop through each gauging station and download the csv file
for id in stn['id']:
    #url of csv file to download
    csv_url = f'https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{id}.csv'
    #download the csv file
    req = requests.get(csv_url) #request the url
    url_content = req.content #get the content of the url
    #save the content of the url to a csv file
    csv_file = open(f'data/gauge_precip/{id}.csv', 'wb') #open a file in write binary mode, this will create the file if it doesn't exist
    csv_file.write(url_content) #write the content of the url to the file

