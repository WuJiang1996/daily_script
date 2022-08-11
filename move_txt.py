import os 
import numpy as np

def mvImgFromDir(dir_):
    ns1 = os.listdir(dir_)
    ns = []
    for n in ns1:
        if ".txt" in n:
            ns.append(os.path.join(dir_,n))
    for n in ns:
        nn = np.loadtxt(n)
        if(len(nn.shape)<2 and nn.shape[0]==0):
            os.system("sudo rm -r {}".format(n))
            os.system("sudo rm -r {}".format(n.replace(".txt",".jpg")))

dir_ = "/data/windows/红外/数据/已标注数据/之前的数据_合并"
mvImgFromDir(dir_)