import numpy as np
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ParticlesSolid, TrailSolid
from bim4loc.maps import RayCastingMap
import time
import logging
import pickle
import os
from bim4loc.evaluation import localization_error, localization_error_2d,

dir_path = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(dir_path, "25a_data.p")
data = pickle.Unpickler(open(file, "rb")).load()
file = os.path.join(dir_path, "25a_results.p")
results = pickle.Unpickler(open(file, "rb")).load()