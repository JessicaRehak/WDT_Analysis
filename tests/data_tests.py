from nose.tools import *
import analysis.core as wdt
import numpy as np

class TestClass:

    @classmethod
    def setup_class(cls):
        filename = './tests/wdt_runs/S0100/W0100/runs/run1_res.m'
        cls.data = wdt.DataFile(filename)

    def test_DataFile_value(self):
        """ Verify DataFile is getting values correctly """
        ana_keff = np.array([[ 1.77253  ,  1.76093  ,  0.0114943]])
        assert_true(np.allclose(ana_keff, self.data.get_data('ANA_KEFF')))

    def test_DataFile_error(self):
        """ Verify DataFile is getting errors correctly """
        ana_keff = np.array([[ 0.00024,  0.0002 ,  0.00431]])
        assert_true(np.allclose(ana_keff, self.data.get_data('ANA_KEFF', err=True)))

    def test_DataFile_reshape_size(self):
        """ Verify matrix data, when reshaped, forms a square matrix of
        the correct size """
        inf_s0 = self.data.get_data('INF_S0',reshape=True)
        assert_true(np.shape(inf_s0) == (11,11))

    def test_DataFile_reshape_vals(self):
        """ Verify values after reshape """
        inf_s0_0 = np.array([ 9.71843000e-02, 5.00055000e-02,
                              8.21536000e-05, 2.94249000e-08, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
        assert_true(np.allclose(inf_s0_0,self.data.get_data('INF_S0',reshape=True)[0]))

    def test_DataFile_reshape_err(self):
        """ Verify errors after reshape """
        inf_s0_0 = np.array([ 2.70000000e-04, 5.60000000e-04,
                              1.54100000e-02, 1.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
        assert_true(np.allclose(inf_s0_0,self.data.get_data('INF_S0',reshape=True, err=True)[0]))

    def test_DataFile_CPU(self):
        """ DataFile gets cpu value correctly """
        assert_true(np.allclose(49.391399999999997, self.data.get_cpu()))
        
    @raises(KeyError)
    def test_DataFile_bad_label(self):
        """ Sending a bad Serpent parameter should return a key error """
        ana_keff = self.data.get_data('WRONG_LABEL')

    def test_DataFile_FOM_of_constant(self):
        """ Requesting the FOM of a constant should return 0 """
        eq_(0, self.data.get_fom('TOT_CPU_TIME'))

    def test_DataFile_FOM_values(self):
        """ DataFile should return correct value when calculating FOM """
        fom = np.array([[ 15578.97788826, 133112.68680858,
                          140210.80099436, 133112.68680858, 147892.18161859,
                          114775.73505434, 140210.80099436, 258245.40387226,
                          1198014.18127723, 277728.93914383, 56240.11017663]])
        ok_(np.allclose(fom, self.data.get_fom('INF_FLX')))

    def test_DataFile_FOM_reshape_shape(self):
        """ DataFile should return the correct shape when calculating
        FOM using reshape """

        eq_((11,11), np.shape(self.data.get_fom('INF_SP0',reshape=True)))
    def test_DataFile_FOM_reshape_values(self):
        """ DataFile should return the correct values when calculating FOM using reshape """
        fom = np.array([ 2.77728939e+05, 6.23159116e+04,
                         8.52596187e+01, 2.02464397e-02, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
        ok_(np.allclose(fom, self.data.get_fom('INF_SP0', reshape=True)[0]))
