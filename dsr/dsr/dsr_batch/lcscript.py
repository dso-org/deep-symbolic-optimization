
import os
import copy


class LCScript:

   def __init__(self,_argfname,_Lfnames,_Lbase,_config):

       self.argfname = _argfname
       self.Lfnames  = _Lfnames
       self.Lbase    = _Lbase
       self.config   = _config
       self.getconffields(['pythonexec','mainpyfile','expdirs','bank','wtimelim'])
       cpath = os.path.splitext(os.path.abspath(__file__))[0]
       cpath = os.path.abspath(os.path.join(cpath, os.pardir))
       self.cpath = os.path.abspath(os.path.join(cpath, os.pardir))
       self.init()

   def getconffields(self,Lfields):
       for p in Lfields:
           setattr(self,p,self.config[p])

   def replacestrs(self,newl,bfnameonly):
        newl = newl.replace("filename",bfnameonly)
        newl = newl.replace("bank",self.bank)
        newl = newl.replace("optdir",self.cpath)
        newl = newl.replace("wtimelim",self.wtimelim)
        return newl 

   def savescript(self,fname,ffname,fnameonly):

       bfnameonly = os.path.join(self.expdirs[fnameonly],fnameonly)
       with open(fname,"w") as f:
            for l in self.Lbase:
                newl = l.replace("cmd",self.pythonexec+" "+self.mainpyfile+" "+ffname)
                newl = self.replacestrs(newl,bfnameonly)
                f.write(newl)

   def savescriptlist(self,fname,ffname,fnameonly):

       bfnameonly = fnameonly      
       with open(fname,"w") as f:
            for l in self.Lbase:
                   if l.find("cmd")>=0:
                      stcmdline = copy.copy(l)   
                      continue
                   newl = self.replacestrs(l,bfnameonly)           
                   f.write(newl)
            for kf in self.Lfnames:
                ffname    = kf[0]
                fnameonly = kf[1]
                fmsubn = os.path.join(self.expdirs[fnameonly],fnameonly)
                outname = ffname +">"+fmsubn+".out"
                newl = stcmdline.replace("cmd",self.pythonexec+" "+self.mainpyfile+" "+outname)
                f.write(newl)
     
   def init(self):

       Lscripts = []
       for f in self.Lfnames:

           fname0 = f[0]
           fname1 = f[1]
           fname = os.path.join(self.expdirs[fname1],fname1)
           self.savescript(fname,fname0,fname1)                    
           Lscripts.append(fname)

       fname = os.path.join(self.config['basepath'],self.argfname)
       self.savescriptlist(fname,fname,fname) 
       os.chmod(fname,0o0700)
       print(" Serial batch: %s"%fname)

       fmainscript = os.path.join(self.config['basepath'],self.config['mainscript']+"_"+self.argfname)
       with open(fmainscript,"w") as fscript:
           for s in Lscripts:
               newl = "msub "+s+"\n" #+">"+scfname+"\n"
               fscript.write(newl)

       os.chmod(fmainscript,0o0700)
       print(" Parallel batch: %s"%fmainscript)


