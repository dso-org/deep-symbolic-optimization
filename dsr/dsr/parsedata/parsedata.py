
import chardet
import pandas as pd
import csv
import copy
from sympy.parsing.sympy_parser import parse_expr

class ParseData:

      def __init__(self,_fdata,_setid):
          self.fdata    = _fdata
          self.setid    = _setid 
          self.objst    = 'Objective Function'
          self.labelvar = 'x'
          self.defdist  = {} #-- the default distribution dictionary {}
          self.types    = {"Variables":int}
          self.dataexpr = {}
          self.dataset  = self.read()

      def read(self):
          with open(self.fdata, 'rb') as f:
               result = chardet.detect(f.read())  # or readline if the file is large
          df = pd.read_csv(self.fdata, encoding=result['encoding'])
#          df.replace({'\'': '"'}, regex=True) 
          for index, row in df.iterrows():
              if row['Dataset'] == self.setid:
                 return row
          return {} 
      #--- old way of reading: NOT used anymore
          with open(self.fdata) as csvfile:
               csvrows = csv.DictReader(csvfile, delimiter=',')
               for row in csvrows:
                   if row['Dataset'] == self.setid:
                      return row
          return {} 

#--- Return types:
#--- 0: no error
#--- 1: data set not found
#--- 2: parsing error
#--- 3: error in training set
#--- 4: error in testing set

      def getexpression(self):
          return self.dataset[self.objst]

      def parse(self):
          if len(self.dataset) == 0:
             return 1

          expr = self.getexpression()          
          try:
             ep = parse_expr(expr)
          except: 
             return 2

          #-- converts the tree to binary
          binexpr = self.converttobinarytree(ep)
          return self.processdict(binexpr)         

      def processdict(self,binexpr): 
          self.dataexpr = {}
          for k,v in self.dataset.items():             
              if isinstance(v, str):
              #-- these characters cause problems depending on the enconding
                 v = v.replace(chr(8221),'"')
                 v = v.replace(chr(8220),'"')
              try:
                  self.dataexpr[k] = eval(v)
              except:
                  self.dataexpr[k] = v
              if k in self.types:
                 self.dataexpr[k] = self.types[k](self.dataexpr[k])
          self.dataexpr["traverse"] = binexpr
          self.VarList = self.getvarlist() 
          ecode = self.updatedist('Training Set',3) #-- the second parameter is the error code
          if ecode:

             return ecode
          return self.updatedist('Testing Set',4)

      def __call__(self):
          return self.dataexpr

      def getvarlist(self):
          Lvars = []
          for i in range(1,self.dataexpr['Variables']):
              nvarname = self.labelvar+str(i)
              Lvars.append(nvarname) 
          return Lvars

      def updatedist(self,kset,ecode):
          ddata = copy.copy(self.dataexpr[kset])
          if ddata is not None:
             if "all" in ddata:
                del self.dataexpr[kset]["all"]
                dist = ddata["all"]
                newddata = {}
                for k in self.VarList:
                    newddata[k] = copy.copy(dist)
                self.dataexpr[kset] = newddata
          else:
                self.dataexpr[kset] = {}

#--- if there is no distribution for the variable k, k will have the default dictionary 
          for k in self.VarList:
              try:
                 if k not in self.dataexpr[kset] or self.dataexpr[kset][k] is None:
                    self.dataexpr[kset][k] = self.defdist
              except:
                 return ecode
         
          return 0

      def converttobinarytree(self,ep):
          expr = self.converttobinary([],ep)
          #-- convert to string
          stexpr = []
          for e in expr:
              stexpr.append(str(e))
          return stexpr

      def converttobinary(self,argnewep,ep):

          if len(ep.args)==0:
             return argnewep

          nexpr = 0   
          rexpr = str(ep.func.__name__)
          newep = []
          newep.append(rexpr)     
    
          for arg in ep.args:
              nexpr += 1
              tmpnewep = self.converttobinary([arg],arg)
              if nexpr <= 2:
                 newep.extend(tmpnewep)
              else:
                 tmpexpr2 = copy.copy(newep)
                 newep = []
                 newep.append(rexpr)
                 newep.extend(tmpnewep)
                 newep.extend(tmpexpr2)

          argnewep.extend(newep)
          return newep


