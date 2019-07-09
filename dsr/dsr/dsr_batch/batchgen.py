
import os
import copy

from .savejson import JSonFile
from .combination import Combination
from .lcscript import LCScript
from .basebatch import DSRBaseBatch
from .dictattr import DictAttr
from .util import configfname, getname, getnamed


class DSRBatch(DSRBaseBatch):

   def __init__(self,_ifile):

       DSRBaseBatch.__init__(self,_ifile)
       dfname = os.path.split(self.configfname)
       fname  = os.path.join(self['basepath'],self.configfname)
       JSonFile(self.ConfigParams).save(fname)
    
   def __call__(self):

       self.batchfilenames = self.basebatchid
       with open(self['basescript']) as f:
            Lbase = f.readlines()
       LCScript(self.batchfilenames,self.generatefiles(),Lbase,self)

   def getaction(self,namepars):
       if namepars in self['actions']:
          return self['actions'][namepars]
       return None

   def setparamval(self,v,ld,lpars,cparams,argk,pval):

       lpars = argk[0]
       if ld:
          del lpars[-1]

       curraction = self.getaction(getnamed(lpars))
       if curraction is not None:
          if 1: #len(curraction)>0:
            if argk[0][len(lpars)-1] not in v:
               v[argk[0][len(lpars)-1]] = {}
            v = v[argk[0][len(lpars)-1]]
            for k,cv in self['actions'][getnamed(lpars)].items():
                if pval not in v:
                   v[pval] = {} 
                v[pval][k] = cv 
            if pval not in v:
               v[pval] = {} 
                 
          else:
            for k in pval.keys():
                if pval[k] == self["stnovar"]:
                   continue
                v[k] = pval[k] 
       else:
            v[lpars[len(lpars)-1]] = pval

       return cparams

   def setcparams(self,cparams,argk,pval):

       v     = cparams
       ld    = 0
       prev  = None
       lpars = copy.copy(argk[0])
       del lpars[-1]

       for p in lpars:
           prev = v
           v = v[p] 
           if p in self.lvar: 
              ld = 1

       if len(lpars)>0:
          cparams = self.setparamval(v,ld,lpars,cparams,argk,pval)

       return cparams

   def printfile(self,name,cparams):

       print(" Saving experiment %s ... "%name,end =" ")
       dirname = self.addbasepath(name)
       self.config['expdirs'][name] = dirname
       fname = os.path.join(dirname,name+self.inputex)
       JSonFile(cparams).save(fname)
       print("Done!")
       return fname

   def prncomb(self,st,Lcomb,rv):

       self.config['expdirs'] = {}
       Lidcomb = []
       Lfnames = []
           
       for c in Lcomb:
           cparams = copy.deepcopy(self.baseparams)
           name    = getname(st,c)
           fname   = self.batchfilenames+"_"+name
           i = 0
           dcomb,idhash= self.getdictcomb(c,rv)
           dcomb['id'] = fname 
           idhash.append('id')
           Lidcomb.append(dcomb)         
           argrv = copy.deepcopy(rv)

           for k in argrv:
               argk        = copy.deepcopy(k)
               cargcparams = copy.deepcopy(self.baseparams)
               cparams     = self.setcparams(cparams,argk,c[i])
               i += 1

           fname  = fname.replace("_"+self['stnovar'],"")
           ffname = self.printfile(fname,cparams)
           Lfnames.append((ffname,fname))

       return Lfnames 

   def getdatalist(self,vdata):

       ldata = []
       for kin, din in vdata.items():
           lddata = FileIO(self,kin).read()
           for d in lddata:
               ldata.append(d[din]) 
       return ldata

   def getlistvals(self,v):

       listvals = []

       try:
          for i in range(v['beg'],v['end'],v['step']):
              listvals.append(i)

       except:
          print("Invalid dictionary entries!")
          exit(0)

       return listvals

   def setparamlists(self):

       stcomb  = "comb"       
       tmpcomb = self.params.copy()
       tmpcomb[stcomb] = {}
       for k1,v1 in self.params[stcomb].items():
         tmpcomb[stcomb][k1] = {}
         for k,v in v1.items():
           if isinstance(v,dict):
              tmpcomb[stcomb][k1][k] = self.getlistvals(v) 
           else:
              tmpcomb[stcomb][k1][k] = copy.copy(v)
       cindex  = 0

       if stcomb in self.params:
          for k,v in tmpcomb[stcomb].items():
              if 1:
               for k2,v2 in v.items():
                 idcomb = stcomb+str(cindex)+str(k)+str(k2)
                 self.params[idcomb] = {}
                 self.params[idcomb][k] = {}
                 ldata = v2
                 if isinstance(v2,dict):
                    ldata = self.getdatalist(v2)
                 self.params[idcomb][k][k2] = ldata

                 cindex += 1
          del self.params[stcomb]

       self.lvarparam = []
       for k,v in self.params.items():
          self.lvarparam.append(DictAttr(v,self.lvar)())

   def generatefiles(self):

       self.setparamlists()
       Lfnames = []
       Lcomb,rv,idval = Combination(self.lvarparam,self)()
       return self.prncomb("",Lcomb,rv)

