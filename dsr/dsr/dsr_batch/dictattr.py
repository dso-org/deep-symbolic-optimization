


class DictAttr:

   def __init__(self,_val,_ladd):
    
       self.val  = _val     #--- this is the batch config file    
       self.ladd = _ladd    #--- this is the base config file

   def __call__(self):

       v     = self.val
       Lkeys = []    
       ldvar = 0

       while isinstance(v,dict):

          dt = v
          for k,kv in dt.items():
              v = kv
              Lkeys.append(k) 
              if k in self.ladd:

                 Ldt = []
                 Lkt = []
                 for kval,vk in v.items():
                     Lv = []                     
                     Lkt.extend(Lkeys)
                     Lkt.append(kval)

                     for k2,v2 in vk.items():
                         Lv.append({k2:v2})

                     v = Lv
                     Ldt.append((Lkt,v,len(v)))                    
                     Lkt = []

                 return Ldt
                 ldvar = 1
              break
          if ldvar:
             break

       return (Lkeys,v,len(v))

