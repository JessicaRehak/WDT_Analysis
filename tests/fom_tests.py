from nose.tools import *
import analysis.core as core
import analysis.fom as fom
import numpy as np
import os

class TestClass:

    @classmethod
    def setup_class(cls):
        cls.base_dir = './tests/fom_data/'
        cls.test_analyzer = fom.Analyzer(cls.base_dir)
        cls.cycles = [10, 20, 30]
        cls.cpu    = np.array([10.5, 20.5, 30.5])
        cls.error1  = np.array([0.00030, 0.00020, 0.00010])
        cls.error2  = np.array([0.00032, 0.00022, 0.00012])

    
    def test_fom_file_upload(self):
        """ FOM Analyzer should upload all the correct files in a directory """
        files = self.test_analyzer.get_filenames()
        file_loc = os.path.abspath(self.base_dir) + '/'
        
        eq_(len(files), 3)
        ok_(all([file_loc + e in files for e in ['res_10.m', 'res_20.m', 'res_30.m']]))
        
    @raises(AssertionError)
    def test_fom_bad_location(self):
        """ FOM Analyzer should return an error if a non-existent folder
        is given """
        base_dir = './wrong_file_name/'
        new_analyzer = fom.Analyzer(base_dir)

    def test_fom_good_locations(self):
        """ FOM Analyzer should work for locations without slash"""
        base_dir = './tests/fom_data'
        new_analyzer = fom.Analyzer(base_dir)

    def test_fom_fom_values(self):
        ans    = np.power(self.cpu * np.power(self.error1, 2), -1)
        func   = self.test_analyzer.fom('TEST_VAL')
        ok_(all([e in func[:,1] for e in ans]))
        ok_(all([c in func[:,0] for c in self.cycles]))

    def test_fom_fom_values_grp(self):
        ans    = np.power(self.cpu * np.power(self.error2, 2), -1)
        func   = self.test_analyzer.fom('TEST_VAL', grp = 2)
        ok_(all([e in func[:,1] for e in ans]))
        ok_(all([c in func[:,0] for c in self.cycles]))

    def test_fom_fom_values_cpu(self):
        ans    = np.power(self.cpu * np.power(self.error1, 2), -1)
        func   = self.test_analyzer.fom('TEST_VAL', cycle = False)
        ok_(all([e in func[:,1] for e in ans]))
        ok_(all([c in func[:,0] for c in self.cpu]))

    def test_fom_fom_length(self):
        eq_(np.shape(self.test_analyzer.fom('TEST_VAL')), (3,2))

    def test_fom_err_values(self):
        func = self.test_analyzer.err('TEST_VAL', grp = 1)
        ok_(all([e in func[:,1] for e in self.error1]))
        ok_(all([c in func[:,0] for c in self.cycles]))

    def test_fom_err_size(self):
        eq_(np.shape(self.test_analyzer.err('TEST_VAL')), (3,2))

    def test_fom_fom_multigroup(self):
        ans1   = np.power(self.cpu * np.power(self.error1, 2), -1)
        ans2   = np.power(self.cpu * np.power(self.error2, 2), -1)
        func   = self.test_analyzer.fom('TEST_VAL', grp=[1,2])
        eq_(np.shape(func), (3,3))
        ok_(all([c in func[:,0] for c in self.cycles]))
        ok_(all([e in func[:,1] for e in ans1]))
        ok_(all([e in func[:,2] for e in ans2]))
