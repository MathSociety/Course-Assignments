#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 10:26:29 2024

@author: yoananya
"""
import matplotlib.pyplot as plt
nframes_values=[1,2,4,6,8,10]
avg_rewards=[48.33,19.28,74,46.17,26.92,50.33]
plt.figure(figsize=(8, 6))
plt.plot(nframes_values, avg_rewards, marker='o')
plt.title('Nframes vs Average Reward', fontsize=14)
plt.xlabel('Nframes', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.grid(True)
plt.show()