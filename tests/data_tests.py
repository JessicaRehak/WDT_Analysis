from nose.tools import *
import analysis.core as wdt
import numpy as np

class TestClass:

    @classmethod
    def setup_class(cls):
        filename = './tests/wdt_runs/S0100/W0100/runs/run1_res.m'
        cls.data = wdt.DataSet(filename)

    def test_DataSet_value(self):
        """ Verify DataSet is getting values correctly """
        ana_keff = np.array([[ 1.77253  ,  1.76093  ,  0.0114943]])
        assert_true(np.allclose(ana_keff, self.data.get_data('ANA_KEFF')))

    def test_DataSet_error(self):
        """ Verify DataSet is getting errors correctly """
        ana_keff = np.array([[ 0.00024,  0.0002 ,  0.00431]])
        assert_true(np.allclose(ana_keff, self.data.get_data('ANA_KEFF', err=True)))

    def test_DataSet_reshape_size(self):
        """ Verify matrix data, when reshaped, forms a square matrix of
        the correct size """
        inf_s0 = self.data.get_data('INF_S0',reshape=True)
        assert_true(np.shape(inf_s0) == (11,11))

    def test_DataSet_reshape_vals(self):
        """ Verify values after reshape """
        inf_s0_0 = np.array([ 9.71843000e-02, 5.00055000e-02,
                              8.21536000e-05, 2.94249000e-08, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
        assert_true(np.allclose(inf_s0_0,self.data.get_data('INF_S0',reshape=True)[0]))

    def test_DataSet_reshape_err(self):
        """ Verify errors after reshape """
        inf_s0_0 = np.array([ 2.70000000e-04, 5.60000000e-04,
                              1.54100000e-02, 1.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
        assert_true(np.allclose(inf_s0_0,self.data.get_data('INF_S0',reshape=True, err=True)[0]))

    def test_DataSet_CPU(self):
        """ DataSet gets cpu value correctly """
        assert_true(np.allclose(49.391399999999997, self.data.get_cpu()))
        
    @raises(KeyError)
    def test_DataSet_bad_label(self):
        """ Sending a bad Serpent parameter should return a key error """
        ana_keff = self.data.get_data('WRONG_LABEL')

    def test_DataSet_FOM_of_constant(self):
        """ Requesting the FOM of a constant should return 0 """
        eq_(0, self.data.get_fom('TOT_CPU_TIME'))
