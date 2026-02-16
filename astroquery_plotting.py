import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from astroquery.eso import Eso
from astropy.io import fits
import time

# 1. QUERY
eso = Eso()
# eso.login('YOUR_USERNAME') 

query = 'HIP 67522' 
instrument = 'HARPS' 

results = eso.query_surveys(target=query, survey=instrument)

if results is None or len(results) == 0:
    raise ValueError(f"No results found for {query}.")

# 2. FILTER FOR SPECTRA 
if "Product category" in results.colnames:
    mask = [("SPECTRUM" in str(x)) for x in results["Product category"]]
    spec = results[mask]
else:
    spec = results

use = spec if len(spec) > 0 else results
arc = str(use["ARCFILE"][0]).strip()

#print(f"Target: {query} | Downloading ARCFILE: {arc}")

# 3. DOWNLOAD FITS FILES

target_dir = r"C:\Users\ilakk\OneDrive\Desktop\astr502\working_dowloads"
timestamp = int(time.time())
filename = f"{query}_{instrument}_{timestamp}.fits"
path = os.path.join(target_dir, filename)
#path = os.path.join(os.path.expanduser("~"), "eso_temp_file1.fits")

#GEMINI AI
try:
    print(f"Attempting manual download to: {path}")
    url = f"https://dataportal.eso.org/dataPortal/file/{arc}"
    response = requests.get(url, stream=True, timeout=30)
    
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download successful.")
    else:
        print(f"Download failed. Status: {response.status_code}")
        print("Note: If status is 401/403, you must use eso.login().")
        path = None
except Exception as e:
    print(f"Error during download: {e}")
    path = None

# 4. OPEN AND PREPARE DATA
if path and os.path.exists(path):
    with fits.open(path) as hdul:
        hdul.info()
        # HARPS 'ADP' files usually store data in Extension 1
        ext = 1 if len(hdul) > 1 else 0
        data_table = hdul[ext].data
        
        # Check if the columns are nested arrays (common in ESO formats)
        if hasattr(data_table['WAVE'][0], '__len__'):
            xvals = data_table['WAVE'][0]
            yvals = data_table['FLUX'][0]
        else:
            xvals = data_table['WAVE']
            yvals = data_table['FLUX']

    xmin, xmax = xvals[0], xvals[-1]
    print(f"Data Loaded. Range: {xmin:.2f} - {xmax:.2f} Å")
else:
    raise FileNotFoundError("FITS file could not be found or created.")

# 5. PLOTTING
# Overall Plot
plt.figure(figsize=(10, 4))
plt.plot(xvals, yvals, color='tab:pink', lw=0.5)
plt.title(f"Full Spectrum: {query} ({instrument})")
plt.xlabel("Wavelength (Å)")
plt.ylabel("Flux")
plt.show()

# Detailed Feature Subplots
fig, graphs = plt.subplots(2, 2, figsize=(12, 10))
plt.suptitle(f'Key Spectral Features: {query}', fontsize=16)

def plot_feature(ax, wave, flux, center, window, title, color):
    # Define the zoom window
    mask = (wave >= center - window) & (wave <= center + window)
    '''
    if not np.any(mask):
        ax.text(0.5, 0.5, f'Feature {center}Å\nNot in Range', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    '''
    ax.plot(wave[mask], flux[mask], color=color)
    ax.axvline(x=center, color='black', ls='--')
    ax.set_title(title)
    
    # Auto-rescale Y-axis with a 10% buffer
    y_seg = flux[mask]
    ymin, ymax = np.nanmin(y_seg), np.nanmax(y_seg)
    pad = 0.1 * (ymax - ymin) if ymax != ymin else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)

# Run the plotting function for your 4 lines
plot_feature(graphs[0,0], xvals, yvals, 6562.8, 50, 'H-alpha (6562.8 Å)', 'tab:pink')
plot_feature(graphs[0,1], xvals, yvals, 6707.8, 10, 'Lithium (6707.8 Å)', 'grey')
plot_feature(graphs[1,0], xvals, yvals, 3933.7, 30, 'Ca II H (3933.7 Å)', 'orange')
plot_feature(graphs[1,1], xvals, yvals, 3968.5, 30, 'Ca II K (3968.5 Å)', 'tab:red')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()