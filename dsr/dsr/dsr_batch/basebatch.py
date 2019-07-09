
import sys
import copy
import json
import os


class DSRBaseBatch:

   def __init__(self,_ifile):

       self.lvar        = ["var"]
       self.inputex     = '.json'
       self.configid    = "config"
       self.configfname = "confbatch.json"  #--- this is the batch config file    
       self.config   = {}
       self.init(_ifile)
       self.readinputbatch()

   def setfilename(self,argv,argfname,ix):

       if len(argv) > ix:
          setattr(self,argfname,argv[ix])

       if self.configfname.find(self.inputex)<0:
          setattr(self,argfname,getattr(self,argfname)+argv[ix])

   def init(self,argv):
       self.setfilename(argv,'configfname',1)

   def __getitem__(self,st):
       return self.config[st]  

   def __setitem__(self,st,val):
       self.config[st] = val
       return self

   def getparams(self):
       if not hasattr(self,'params'):
          return
       self.config.update(self.params[self.configid].copy())
       del self.params[self.configid]      

   def getvals(self,module_name):
       module = sys.modules[module_name]
       par    = {}
       if module:
          par = {key: value for key, value in module.__dict__.items() if not (key.startswith('__') or key.startswith('_'))}
       return par

   def getcurrmodule(self):
       module = sys.modules[__name__]
       par    = {}
       if module:
          par = {key: value for key, value in module.__dict__.items() }
       return par

   def setdefparams(self):
       pk = self.getcurrmodule()['__package__']
       constmodulename = pk+'.default'
       __import__(constmodulename)
       self.config = self.getvals(constmodulename)

   def getdictcomb(self,idcomb,rv):
           dIdComb = {}
           i = 0
           Lidhash = []
           for v in idcomb:
               idhash = self.getstrid(rv[i][0])
               dIdComb[idhash] = v
               i += 1           
               Lidhash.append(idhash) 
           return dIdComb,Lidhash

   def getstrid(self,L):
       strid = L[0]
       for i in range(1,len(L)):
           strid += "_" +L[i]
       return  strid

   def checkdir(self,st,create=0):
       if not os.path.exists(st):
          os.makedirs(st)

   def setbasepath(self):

       self.checkdir(self['basepath'])  
       base_exp = os.path.join(self['basepath'],self['label'])
       self.checkdir(base_exp)  
       base_exp = os.path.join(base_exp,self.baseparamfname)

       self.checkdir(base_exp)  
       self['basepath'] = base_exp
       print("\n Base directory of experiments: %s"%self['basepath'])

   def addbasepath(self,name):
       newdir = os.path.join(self['basepath'],name)
       self.checkdir(newdir)
       return newdir

   def setconfigparams(self):
       self.setdefparams()
       self.getparams()
       self.setbasepath()

   def getjson(self,argfile):
        
       with open(argfile) as f:
            jsonparams = json.load(f)
       return jsonparams

   def setattrparams(self):
       for k,v in self.config.items():
           setattr(self,k,v)

   def readinputbatch(self):

       self.params     = self.getjson(self.configfname)
       self.baseparams = self.getjson(self.params["basedata"]) #self.params["basedata"]
       del self.params["basedata"]
       self.basebatchid = os.path.basename(os.path.splitext(self.configfname)[0])
       self.baseparamfname = os.path.basename(self.basebatchid)
 
       self.setconfigparams()
       self.ConfigParams = self.params
       self.setattrparams()


