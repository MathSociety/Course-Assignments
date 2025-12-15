import numpy as np
import scipy
import matplotlib.pyplot as plt

def compute_SPNE(piefg):
    pass


piefg = []
piefg.append(['P1', 'P2'])
piefg.append('V0')

piefg.append(['V0', 'P1', {'V1':'Stop', 'V2':'Go'}])
piefg.append(['V1', 'P2', {'V3':'Ready', 'V4':'Go', 'V5':'Start'}])
piefg.append(['V2', 0, {'V6':0.6, 'V7':0.4}])
piefg.append(['V6', 'P2', {'V8':'E', 'V9':'F'}])
piefg.append(['V9', 0, {'V10':0.3, 'V11':0.7}])
piefg.append(['V11', 'P1', {'V12':0, 'V13':1}])

piefg.append(['V3', [3, 8]])
piefg.append(['V4', [8, 10]])
piefg.append(['V5', [6, 3]])
piefg.append(['V7', [5, 5]])
piefg.append(['V8', [5, 5]])
piefg.append(['V10', [6, 7]])
piefg.append(['V12', [2, 3]])
piefg.append(['V13', [1, 0]])


spne = compute_SPNE(piefg)