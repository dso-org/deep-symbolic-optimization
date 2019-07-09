
import os
import json


class JSonFile:

    def __init__(self,_data):
        self.data = _data

    def save(self,fname):
        
        exclude = ["load"]
        self.JsonParams = {}
        for p,v in self.data.items():
            self.JsonParams[p] = v

        with open(fname, 'w') as ofile:
             np = 0
             ofile.write(json.dumps(self.JsonParams,indent=3))
             return


