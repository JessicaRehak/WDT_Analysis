"""

.. module:: fom
     :synopsis: Tools for analyzing FOM convergence

.. moduleauthor:: Joshua Rehak <jsrehak@berkeley.edu>

"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import math
import pandas as pd
import core

class Analyzer():
    """ An object containing multiple :class:`analysis.core.DataSet`
    objects with methods to analyze FOM convergence properties. All
    `res.m` files in a directory will be ingested when initialized,
    the intention is that each of these represents the same simulation
    at different cycle values.

    :param location: folder where the Serpent output files are located
    :type location: string

    """
    pass
