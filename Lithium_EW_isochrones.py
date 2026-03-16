
#import EAGLES (Jeffries et. al)
import sys
sys.path.append(r'C:\Users\ilakk\OneDrive\Desktop\astr502\eagles')
from eagles import AT2EWm, eAT2EWm

import numpy as np
import matplotlib.pyplot as plt

ages = [5, 10, 15, 20, 30, 50, 100, 300, 1000, 5000]
plot_dispersion = False
tstep = 0.002

fig, ax = plt.subplots()
plt.xlabel('Teff (K)')
plt.ylabel('LiEW (mA)')
ax.set_xlim(6500, 3000)

lteff = np.arange(3.4772, 3.8130, tstep)

for t in ages:
    lAge = np.log10(t) + 6
    ewm  = AT2EWm(lteff, lAge)   # direct function call, no model object
    eewm = eAT2EWm(lteff, lAge)

    name = 'iso_' + str(t) + '.txt'
    np.savetxt(name, np.column_stack((10**lteff, ewm, eewm)),
               fmt='%.1f %.1f %.1f', delimiter=' ',
               header="Teff(K) EWLim(mA) eEWLi(mA)")

    ax.plot(10**lteff, ewm, label='%s Myr' % t)

    if plot_dispersion:
        plt.fill_between(10**lteff, ewm - eewm, ewm + eewm, alpha=0.3)

if len(ages) < 5:
    plt.legend()
else:
    ax.text(0.01, 0.95, str(ages) + " Myr", transform=ax.transAxes)

plt.show()

import os
print(os.listdir(r'C:\Users\ilakk\OneDrive\Desktop\astr502\eagles'))