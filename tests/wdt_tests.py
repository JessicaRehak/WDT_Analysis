from nose.tools import *
import analysis.core as core
import analysis.wdt as wdt
import numpy as np
import os

class TestClass:

    @classmethod
    def setup_class(cls):
        # Location of the test data
        cls.base_dir = './tests/fom_data'
        cls.test_run = wdt.SerpentRun(cls.base_dir)
        # This is the actual data contained in the fom_data directory,
        # already sorted by cycles
        cls.cycles = np.array([10, 20, 30])
        cls.cpu    = np.array([10.5, 20.5, 30.5])
        #  TEST_VAL
        ## Group 1
        cls.val1    = np.array([1.23456e02]*3, ) 
        cls.error1  = np.array([0.00030, 0.00020, 0.00010])
        cls.val2    = np.array([2.23456e02]*3) 
        cls.error2  = np.array([0.00032, 0.00022, 0.00012])
        cls.materror11 = np.array([ 0.00030, 0.00020, 0.00010])
        cls.materror12 = np.array([ 0.00032, 0.00022, 0.00012])
        cls.materror21 = np.array([ 0.00034, 0.00024, 0.00014])
        cls.materror22 = np.array([ 0.00036, 0.00026, 0.00016])

    ## ==================== INIT =====================================
        
    @raises(AssertionError)
    def test_init_filename(self):
        """ SerpentRun init should throw AssertionError if non-existent folder is given """
        base_dir = './bad_file/'
        bad_run = wdt.SerpentRun(base_dir)

    @raises(AssertionError)
    def test_init_no_m_files(self):
        """ SerpentRun init should throw AssertionError if no m-files in folder """
        base_dir = '.'
        bad_run = wdt.SerpentRun(base_dir)

    def test_init_upload(self):
        """ SerpentRun init should upload all correct files in a directory """
        
        eq_(len(self.test_run.files), 3)         # 3 files uploaded
        
        filenames = [file.filename for file in self.test_run.files]
        file_loc = os.path.abspath(self.base_dir) + '/'
        ok_(all([file_loc + e in filenames for e in
                 ['res_10.m','res_20.m','res_30.m']])) # Correct filenames

    def test_init_sorting(self):
        """ SerpentRun init should sort all files based on cycle number """

        cycles = np.array([file.cycles for file in self.test_run.files])
        ok_(np.all(cycles == self.cycles))

    @raises(AssertionError)
    def test_init_params(self):
        """ SerpentRun init should throw assertion error if params
        are not a tuple or list of tuples """
        bad_run = wdt.SerpentRun(self.base_dir, params=[1,("str",1)])

    def test_init_param_list(self):
        """ SerpentRun init should cast params into a list """
        good_run = wdt.SerpentRun(self.base_dir, params=('test',0.1))
        ok_(isinstance(good_run.params, list))

    def test_init_cycles(self):
        """ SerpentRun init should create cycles list with proper values """
        ok_(np.all(self.test_run.cycles == self.cycles))

    def test_init_cpus(self):
        """ SerpentRun init should create cycles list with proper values """
        ok_(np.all(self.test_run.cpus == self.cpu))


    ## ==================== GET_DATA =================================
    
    def test_get_data(self):
        """ SerpentRun get_data should return the correct error and
        values """

        ok_(np.all(self.error1 == self.test_run.get_error('TEST_VAL',1))) # Err grp 1
        ok_(np.all(self.error2 == self.test_run.get_error('TEST_VAL',2))) # Err grp 2

    ## ==================== FOM =================================

    def test_calc_fom_cpu(self):
        """ SerpentRun calc_fom should return correct fom using cpu time """
        fom = np.power(np.multiply(np.power(self.error1,2), self.cpu),-1)
        ok_(np.all(fom == self.test_run.fom('TEST_VAL',1)))

    def test_calc_fom_cycles(self):
        """ SerpentRun calc fom should return correct fom using cycle time """
        fom = np.power(np.multiply(np.power(self.error1,2), self.cycles),-1)
        ok_(np.all(fom == self.test_run.fom('TEST_VAL',1, cpu=False)))

    def test_calc_fom_cap(self):
        """ SerpentRun calc_fom should return correct fom when capped """
        fom = np.power(np.multiply(np.power(self.error1,2), self.cpu),-1)[:2]
        ok_(np.all(fom == self.test_run.fom('TEST_VAL',1, cap=25)))
