
import copy
import math

class Combination:

   def __init__(self,_Lparams,_config):
       self.Lparams = _Lparams
       self.config  = _config
       self.init()

   def init(self):

       self.Lv    = copy.copy(self.Lparams) 
       self.idval = 1 

   def __call__(self):

       rv   = sorted(self.Lv,key=lambda x: x[2],reverse=True)
       tot  = 1
       Lmax = []

       for c in rv:
           tot *= c[2]  
           Lmax.append(c[2]) 

       Lc = []
       for n in range(tot):
           Lc.append(self.getval(n,Lmax,tot))
   
       Lvals = []
       for c in Lc:
           idcomb = self.getvals(c,rv)
           Lvals.append(idcomb)

       return Lvals,rv,self.idval

   def getval(self,n,Lmax,tot):

       vc   = [ 0 for i in range(len(Lmax))]
       i    = 0
       num  = n
       ntot = num
       ix   = 0 
       Ltot = [ 0 for i in range(len(Lmax))]
       last = None
       pos1 = 0
       num = n

       for i in range(len(Lmax)):
           vc[i] = (num % Lmax[i])
           num = int(math.floor(num/Lmax[i]))

       return vc

   def getvals(self,c,rv):

       Lvars = []
       i     = 0    
       for r in rv:
           Lvars.append(r[1][c[i]])
           i += 1

       return Lvars


