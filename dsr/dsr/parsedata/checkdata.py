
import csv

#-- this class can be used to check the data sets, i.e., if they can be parsed correctly 
from parsedata import ParseData

class CheckData:

#-- fout is a csv file. If fout is provided, the checks will be saved to a file
      def __init__(self,_fdata,_fout=""):

          self.fdata = _fdata
          self.fout  = _fout
          Ldata      = self.read()
          Lchecks    = self.check(Ldata)
          self.write(Lchecks)

      def read(self):
         
          Ldata = []
          with open(self.fdata) as csvfile:
               csvrows = csv.DictReader(csvfile, delimiter=',')
               for row in csvrows:
                   Ldata.append(row)
          return Ldata

      def write(self,Lchecks):
          if len(self.fout) == 0:
             for c in Lchecks:
                 print(" Dataset: %15s    Check result: %2d"%(c['Dataset'],c['Result']))
             return
                   
          with open(self.fout, 'w') as csvfile:
               writer = csv.DictWriter(csvfile,fieldnames=['Dataset','Result'])
               writer.writeheader()
               for dt in Lchecks:
                   writer.writerow(dt)

      def check(self,Ldata):

          if len(Ldata) == 0:
             print("No data sets were retrived!")
             exit(0)

          Lchecks = []
          for dt in Ldata:
              data = ParseData(self.fdata,dt['Dataset'])
              result = data.parse()
              Lchecks.append({'Dataset':dt['Dataset'],'Result':result})
          
          return Lchecks


