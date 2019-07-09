
import csv
import os

class FileIO:

      def __init__(self,_fname,_header=None):
          self.fname  = _fname
          self.header  = _header
          self.writeh()

      def read(self):

          Lread = []
          if not os.path.isfile(self.fname):
             print(" File %s not found!"%self.fname)
             return Lread

          with open(self.fname,'r') as csvfile:
               reader = csv.DictReader(csvfile, delimiter=',')
               for row in reader:
                   Lread.append(row) 

          return Lread

      def createheader(self,fname,writer):
          writer.writeheader()

      def writerow(self,fname,writer,dt):
          writer.writerow(dt)

#--- read header
      def readh(self):
        with open(self.fname) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                return row.keys()

#--- write header
      def writeh(self):

          with open(self.fname, 'w') as csvfile:
               writer = csv.DictWriter(csvfile,fieldnames=self.header)
               self.createheader(self.fname,writer)

      def writeupdate(self,field,Ldata,header=None,cpfile=""):
          if os.path.isfile(self.fname):
             Lddict = FileIO(self.fname).read()
             dtlist = copy.copy(Ldata)
             for d in Lddict:
                 copydt = 1
                 for dt in Ldata:
                     if dt[field] == d[field]:
                        copydt = 0
                        break;
                 if copydt:
                    dtlist.append(d) 
             Ldata = dtlist

          self.write(Ldata,header,cpfile)
           

      def write(self,Ldata,header=None,cpfile=""):
          if len(Ldata) == 0:
             return

          if header is None:
             header = Ldata[0].keys() 

          with open(self.fname, 'w') as csvfile:
               writer = csv.DictWriter(csvfile,fieldnames=header)
               self.createheader(self.fname,writer)
               for d in Ldata:
                   self.writerow(self.fname,writer,d)

#--- update file contents: insert new row
      def update(self,data):

          if self.header is None:
             print(" No header provided! ")
             exit(0)
          with open(self.fname, 'a') as csvfile:
               writer = csv.DictWriter(csvfile,fieldnames=self.header)
               self.writerow(self.fname,writer,data)

      def updateh(self,OutDataList,obj,x,update):

          mode       = self.filemode[update]
          fieldnames = []
          OutputData = {}

          for p in OutDataList:
              OutputData[p] = getattr(obj,p)
              fieldnames.append(p)

          for p in obj.pvarorder:
              OutputData[p] = x[p]
              fieldnames.append(p)

          with open(self.fname, mode) as csvfile:
               writer = csv.DictWriter(csvfile,fieldnames=fieldnames,lineterminator=self.EOL)
               if mode == self.CreateFile:
                  self.createheader(self.fname,writer)
               self.writerow(self.fname,writer,OutputData)


