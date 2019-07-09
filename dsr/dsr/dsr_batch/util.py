


ftype = '.csv'

def _ftype(fname):
    return fname+ftype

schar = [":","\\"," / ",",","(",")"," ","[","]","&","'","%"," "]

def configfname(fname):

          sname = fname
          for c in schar:
              sname = sname.replace(c,"_")
          return sname


def configname(env,fname,argchars=[]):

          scharname = schar.copy()
          if len(argchars)>0:
             scharname.extend(argchars)
          sname = fname
          for c in scharname:
              sname = sname.replace(c,"_")
          return sname

def getname(st,args):

       name = st
       for s in args:
           if len(name)>0:
               name +="_"
           cs = s
           if isinstance(s,dict):
               cs = list(s.keys())[0]
           name += str(cs)

       name = configfname(name)
       return name

def getnamed(lst):

       name = ""
       f = 1
       for s in lst:
           if not f:
              name += "_" 
           name += s
           f = 0
       return name


