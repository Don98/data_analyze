import json  
import urllib.request 
import requests

def geoGrab():
    j=0
    f=open(r'Restaurant_Data_Beijing.txt','w') 
    for j in range (0,20):
        part1 = 'http://api.map.baidu.com/place/v2/search?q=%E9%A5%AD%E5%BA%97&page_size=20&page_num='
        part2 = '&region=%E5%8C%97%E4%BA%AC&output=json&ak=YOU_KEYS'#The key you acquired from baidu ,here is the url "http://lbsyun.baidu.com/index.php?title=webapi/guide/webservice-placeapi" 
        part3 =str(j)
        url=part1 + part2 + part3
        j=j+1
        temp=requests.get(url)
        hjson = json.loads(temp.text)
        i=0
        for i in range (0,20):
            lat=hjson['results'][i]['location']['lat']
            lng=hjson['results'][i]['location']['lng']
            print ('%s\t%f\t' % (lat,lng))
            f.write('%s\t%f\t\n' % (lat,lng))
            i=i+1
    f.close()
if __name__ == '__main__':
  geoGrab()
