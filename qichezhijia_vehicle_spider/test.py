import requests
from lxml import etree
import re,time,os
class carDownload:
    def __init__(self):
        self.header={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36'}
    def getCarUrl(self,url):
        content=requests.get(url,headers=self.header).text
        dom=etree.HTML(content)
        urls=dom.xpath('//@href')
        page_urls=[url]
        for i in urls:
            if '/pic/series/614-1-p' in i:
                page_urls.append('https://car.autohome.com.cn/pic'+i)
        return list(set(page_urls))
    def getCarPicUrl(self,url):
        content=requests.get(url,headers=self.header).text
        dom=etree.HTML(content)
        urls=dom.xpath('//@href')
        page_urls=[]
        for i in urls:
            if '/photo/series/' in i:
                page_urls.append('https://car.autohome.com.cn/'+i)
        return list(set(page_urls))
    def getPic(self,name,url):
        content=requests.get(url,headers=self.header).text
        dom=etree.HTML(content)
        urls=dom.xpath('//*[@id="main"]/div[1]/img/@src')[0]
        with open(f'./{name}/'+str(time.time())+'.jpg','wb') as f:
            f.write(requests.get('https:'+urls).content)
            print("保存成功")
    def main(self,path,url):
        if not os.path.exists(path):
            os.mkdir(path)
        carPage=self.getCarUrl(url)
        for i in carPage:
            for n in self.getCarPicUrl(i):
                self.getPic(path,n)

car=carDownload()
car.main('思域','https://car.autohome.com.cn/pic/series/164-1-p1.html')