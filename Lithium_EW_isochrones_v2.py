# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:02:16 2026

@author: ilakk
"""

import zipfile, os
os.chdir(r'C:\Users\ilakk\OneDrive\Desktop\astr502\eagles') 
zip_path = r'C:\Users\ilakk\OneDrive\Desktop\astr502\eagles\eaglesv2_0.zip'
extract_to = r'C:\Users\ilakk\OneDrive\Desktop\astr502\eagles'
'''
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg' if this doesn't work
'''
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_to)

# Confirm it extracted correctly
print(os.listdir(extract_to))

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 12:41:08 2022

@author: Rob Jeffries, George Weaver

Produces and plots a set of isochrones of EWLi vs Teff (and the modelled dispersion)
for a defined list of ages; and saves them as individual text files.
Optionally it can plot the model dispersion around each isochrone.

"""

import numpy as np
import matplotlib.pyplot as plt
# import the EWLi prediction model from the main EAGLES code
from eaglesv2_0 import EWLi_Model




#choose whether to use the ML model
use_ML = True
#Instantiate the model
model = EWLi_Model(use_ML)


# Modify the list below for the isochrones you want to produce (units are Myr)
ages = [5, 10, 15, 20, 30, 50, 100, 300, 1000, 5000]

# e.g. for fewer isochrones
#ages = [10, 100, 1000]

# Modify this flag to true if you want the plot to include the dispersion
# looks messy if there are many closely spaced isochrones
plot_dispersion = False

# The step in logarithmic temperature
tstep = 0.002

# parameters for the plot

fig, ax = plt.subplots()
plt.xlabel('Teff (K)')
plt.ylabel('LiEW (mA)')
ax.set_xlim(6500, 3000)

# set up a an equally spaced set of log temperatures between 3000 and 6500 K
lteff = np.arange(3.4772, 3.8130, tstep)

# loop over the ages
for t in ages :

    lAge = np.log10(t)+6  # log age in years
    ewm = model.get_EWm(lteff, lAge)
    eewm = model.get_eEWm(lteff, lAge)

    # save the results as a simple .txt file    
    name = 'iso_'+str(t)+'v2_0.txt'
    np.savetxt(name, np.column_stack((10**lteff, ewm, eewm)), fmt='%.1f %.1f %.1f', delimiter=' ', header = "Teff(K) EWLim(mA) eEWLi(mA)")
 
    ax.plot(10**lteff, ewm, label='%s Myr' %t)

    # if the plot_dispersion flag then shade the dispersion region
    # looks quite messy if there are lots of isochrones because of the overlap
    if plot_dispersion :
        plt.fill_between(10**lteff, ewm-eewm, ewm+eewm, alpha=0.3)

# For the default list of ages and plot_dispersion = False, this is Fig.2 from the paper    

if len(ages) <5 :
    plt.legend()
else :
    ax.text(0.01,0.95, str(ages)+" Myr", transform=ax.transAxes)
plt.title("EAGLES Lithium Equivalent Width Isochrones")


#import CSV
import pandas as pd
df = pd.read_csv(r"C:\Users\ilakk\OneDrive\Desktop\astr502\astr502_spring2026\ASTR502_Master_Parameters_List.csv")

# Confirm the column names
print(df.columns.tolist())
t_eff = df['tic_teff']
li_ew= df['Li_EW']

#plt.plot(t_eff,li_ew,'.')

newdf = df[['hostname', 'tic_teff', 'Li_EW']].dropna()
print(newdf)
# Overlay data points on the same axes as the isochrones
ax.plot(t_eff, li_ew, '.', label='Observed Stars', color='black')

# Now show the combined plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()