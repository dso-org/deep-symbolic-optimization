import unittest
import sys
import os
import numpy as np
import pickle
import sys
import time

def std_error_print(message,exception):
    print("\n======================================================================")
    print("ERROR: {}".format(message))
    print("----------------------------------------------------------------------\n")
    
    raise exception
    

class TestBase(unittest.TestCase):
    r'''
        This is a base class for running test cases. The idea is that all test cases are bases on files in a 
        directory with a name that corresponds to the name of a the method/function to be tested. So long
        as the naming convention is honored, then this is easy to set up for any case. 
    '''
    
    # ******************************************************************************************************************* 
    def __init__(self, methodName = 'runTest', data_dir = None, data_extention = None, timer_loops = 0):
        
        assert(data_dir is not None)
        assert(isinstance(data_dir,str))
        assert(isinstance(methodName,str))
        assert(isinstance(timer_loops,int))
        assert(data_extention is None or isinstance(data_extention,str))
            
        super(TestBase, self).__init__(methodName)
        
        r'''
            Take all the case files in the case file directory and stuff their name into
            a dictionary.
        ''' 

        self.data_dir           = data_dir
        self.data_extention     = data_extention
        
        # Get a listing of all sub-directories in which we place test cases for each method
        self.dirs               = os.listdir(self.data_dir)
        
        # Weed out things which are not directories like tmp files and other weirdness that sometimes just show up places
        self.dirs               = [ d for d in self.dirs if os.path.isdir(os.path.join(data_dir,d)) ] 
        
        # Get the directory names into a dictionary along with the path. 
        if self.data_extention is None:
            self.cases              = { d : os.listdir(os.path.join(self.data_dir,d)) for d in self.dirs }
        else:
            # In this case, we check the file extention and only include if it matches
            self.cases              = { d : [f for f in os.listdir(os.path.join(self.data_dir,d)) 
                                             if f.split('.')[-1] == self.data_extention] for d in self.dirs}
        
        if not self.cases:
            std_error_print("No test cases found in \"{}\"!".format(self.data_dir), FileNotFoundError)
        
        self.timer_loops        = timer_loops
        
        self.timer_slices       = 0
        self.timer_cumulative   = 0.0
        
        self.timer_start        = None
        self.timer_end          = None
        self.timer_elapsed      = None
    
    # ******************************************************************************************************************* 
    def get_cases(self):
        
        r'''
            Test cases are assumed to be in a directory denoted by the calling method name. So for instance,
            
                def test_result_execute(self):
            
            This would then be split with the directory "execute" being the one with the test cases. So,
            to make all this work, one just has to follow the naming convention for unit test method names.
        '''
        
        caller = sys._getframe(1).f_code.co_name
        case   = caller.split('_')[-1]
        
        print("\nRUNNING CASES \"{}\"".format(caller))
        
        case_dir = os.path.join(self.data_dir, case)
        
        return self.cases[case], case_dir
    
    # ******************************************************************************************************************* 
    def timer_block_start(self):
        self.timer_start        = time.time()
        
    # ******************************************************************************************************************* 
    def timer_block_stop(self):
        self.timer_end          = time.time()
        self.timer_elapsed      = self.timer_end - self.timer_start
        self.timer_cumulative   += self.timer_elapsed
        self.timer_slices       += 1
        
    # ******************************************************************************************************************* 
    def timer_show_results(self):
        
        caller                  = sys._getframe(1).f_code.co_name
        
        if self.timer_slices > 0:
            self.timer_average      = self.timer_cumulative / float(self.timer_slices)
            print("TIMER ({}): Total {:.4f} sec, Avg {:.4f} sec, Slices {}, Loops {}".format(caller,self.timer_cumulative,self.timer_average,self.timer_slices,self.timer_loops))
        else:
            std_error_print("No timer slices were recorded by the time computed results", ZeroDivisionError)

        
        

if __name__ == '__main__':
    std_error_print("Cannot call the base class directly!", NotImplementedError)
