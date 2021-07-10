import unittest
import sys
import os
import numpy as np
import pickle
import sys
from dso.program import Program
from test_base import TestBase

# For profiling
import pstats, cProfile
import pyximport

class TestProgram(TestBase):
    
    # ******************************************************************************************************************* 
    def __init__(self, methodName = 'runTest', data_dir = 'data/program/', data_extention='pickle', timer_loops=50):
        super(TestProgram, self).__init__(methodName, data_dir=data_dir, data_extention=data_extention, timer_loops=timer_loops)

    # ******************************************************************************************************************* 
    def test_time_execute(self):
        
        cases, case_dir = self.get_cases()
        
        for i in range(self.timer_loops):
            for c in cases:
                with self.subTest(c=c):
    
                    c_file              = os.path.join(case_dir,c)
                    f                   = open(c_file,'rb')
                    c_data              = pickle.load(f)
                    f.close()
                    
                    Program.library     = c_data['library']
                    Program.const_token = c_data['const_token']
                    Program.X_train     = c_data['X']
                    
                    self.timer_block_start()
                    p                   = Program(c_data['tokens'],False)
                    p.traversal         = c_data['traversal'] # Get rid of this line somehow....
                    self.timer_block_stop()
                    
                    # Simluate 10 optimization steps
                    for i in range(10):
                        self.timer_block_start()
                        result              = p.execute(Program.X_train)
                        self.timer_block_stop()
                
        self.timer_show_results()        
        
    # ******************************************************************************************************************* 
    def test_result_execute(self):
        
        cases, case_dir = self.get_cases()
        
        for c in cases:
            with self.subTest(c=c):
                
                c_file              = os.path.join(case_dir,c)
                
                sys.stdout.write("\t case ... {}".format(c_file))
                
                f                   = open(c_file,'rb')
                c_data              = pickle.load(f)
                f.close()
                
                Program.library     = c_data['library']
                Program.const_token = c_data['const_token']
                Program.X_train     = c_data['X']
                
                p                   = Program(c_data['tokens'],False)
                p.traversal         = c_data['traversal'] # Get rid of this line somehow....
                
                result              = p.execute(Program.X_train)
                
                err                 = np.sum(np.abs(result - c_data['intermediate_result']))
                        
                if not self.assertAlmostEqual(err, 0):
                    sys.stdout.write(" ... OK\n")
                else:
                    sys.stdout.write(" ... FAIL\n")
            
if __name__ == '__main__':
    unittest.main()
    
    # to profile using cProfile, uncomment
    '''
    pyximport.install()
    cProfile.runctx("unittest.main()", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("cumtime").print_stats(25)
    s.print_callees('execute')
    '''
    