from nose.tools import *
import analysis.core as wdt
import numpy as np

class TestClass:

    @classmethod
    def setup_class(cls):
        base_dir = './tests/'
        st_vals = [0.1, 0.075]
        wdt_vals = [0.1]
        cls.test_analyzer = wdt.Analyzer(st_vals, wdt_vals, base_dir)
    
    def setup(self):
        pass
#        base_dir = './tests/'
#        st_vals = [0.1, 0.075]
#        wdt_vals = [0.1]
#       self.test_analyzer = wdt.Analyzer(st_vals, wdt_vals, base_dir)

    def teardown(cls):
        pass
    
    def test_Analysis_data_frames_mean(self):
        # Mean
        inf_flx = np.array([[ 1.00000000e-01, 1.00000000e-01,
                              3.88516046e+04, 1.44060694e+05, 1.66542714e+05,
                              1.79609881e+05, 2.29634674e+05, 1.73867504e+05,
                              2.09243651e+05, 3.10828431e+05, 3.39303873e+05,
                              9.94613565e+05, 9.46112620e+04, 2.88116785e+06,
                              1.00000000e+00], [ 7.50000000e-02, 1.00000000e-01,
                                                 2.86914263e+04, 4.05396961e+05, 9.21662778e+05,
                                                 5.99653306e+05, 8.76070979e+05, 4.51642576e+05,
                                                 2.95865769e+05, 3.20460223e+05, 7.34638279e+05,
                                                 2.97807490e+05, 1.91276486e+05, 5.12316627e+06,
                                                 1.77815613e+00]])
        
        assert_true(np.allclose(inf_flx,self.test_analyzer.data_frame('INF_FLX',rel=False,
                                                                      mean=True,style=False).as_matrix()))
    
    def test_Analysis_data_frames_mean_rel(self):
        # Mean
        inf_flx = np.array([[0.1,0.1,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.  ,
                             0.  , 0.  ], [ 0.075 , 0.1 , -0.26151245,
                                            1.81407059, 4.53409248, 2.33864319,
                                            2.815064 , 1.59762501, 0.41397728,
                                            0.03098749, 1.16513378, -0.7005797 ,
                                            1.02170949, 14.76921115]])
        
        assert_true(np.allclose(inf_flx, self.test_analyzer.data_frame('INF_FLX',rel=True,
                                            mean=True,style=False).as_matrix()))

    def test_Analysis_data_frames_stdev(self):
        inf_flx = np.array([[ 1.00000000e-01, 1.00000000e-01,
                              4.08963692e+04, 5.56068888e+04, 9.62992238e+04,
                              1.02555875e+05, 2.00977978e+05, 9.90797489e+04,
                              2.31251046e+05, 3.13652789e+05, 3.13981086e+05,
                              1.75210549e+06, 8.40807958e+04, 3.29048729e+06,
                              1.00000000e+00], [ 7.50000000e-02, 1.00000000e-01,
                                                 1.72600765e+04, 5.35495747e+05, 2.15509073e+06,
                                                 1.20037329e+06, 1.58902157e+06, 5.97951083e+05,
                                                 3.56941238e+05, 1.98267608e+05, 6.26077204e+05,
                                                 1.63630052e+05, 1.74524251e+05, 7.61463285e+06,
                                                 2.31413532e+00]])
        assert_true(np.allclose(inf_flx, self.test_analyzer.data_frame('INF_FLX',rel=False,mean=False,style=False).as_matrix()))

    def test_Analysis_data_frames_stdev_rel(self):
        inf_flx = np.array([[ 0.1 , 0.1 , 0.  , 0.  , 0.  , 0.  , 0.
                              , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ], [ 0.075 , 0.1 ,
                                                                             -0.57795577, 8.63002532, 21.379108 , 10.70457853,
                                                                             6.90644622, 5.03504843, 0.54352269, -0.36787552, 0.99399656,
                                                                             -0.90660948, 1.07567316, 53.41595814]])

        assert_true(np.allclose(inf_flx, self.test_analyzer.data_frame('INF_FLX',rel=True,mean=False,style=False).as_matrix()))

    def test_Analysis_histogram(self):
        histogram = np.array([[3],[3]])
        assert_true(np.all(histogram == self.test_analyzer.histogram(['INF_FLX','INF_TOT','INF_ABS'],mean=True).as_matrix()))

    def test_Analysis_histogram_stdev(self):
        histogram = np.array([[3],[3]])
        assert_true(np.all(histogram == self.test_analyzer.histogram(['INF_FLX','INF_TOT','INF_ABS'],mean=False).as_matrix()))

    @raises(AssertionError)
    def test_Analysis_plot_mat(self):
        param_sets = range(11)
        self.test_analyzer.plot_mat('INF_SP0', param_sets)
