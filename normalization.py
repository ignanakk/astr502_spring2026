# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:45:04 2026

@author: ilakk
"""
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

query = 'EPIC 211822797' 
instrument = 'HARPS' 

results = eso.query_surveys(target=query, surveys=instrument)

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
        Header=hdul[1].header
        print(Header)
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

print(instrument)

#normalization (from simran)

from specutils import Spectrum, SpectralRegion
from specutils.manipulation import extract_region
import astropy.units as u
from astropy.modeling import models, fitting



spectrum = Spectrum(
    spectral_axis=xvals * u.AA,
    flux=yvals * u.Unit('erg / (cm^2 s AA)')
)

def local_normalize(spectrum, line_center, window=20, shoulder=5):
    """
    Locally normalize a spectrum around a single spectral line.

    Parameters:
        spectrum    : Spectrum object
        line_center : float, central wavelength in Angstroms
        window      : float, half-width of region to extract (default ±20 Å)
        shoulder    : float, width of continuum shoulder on each side (default ±5 Å)

    Returns:
        wave_norm   : wavelength array (Å)
        flux_norm   : normalized flux (continuum ~ 1.0, absorption lines dip below)
    """
    lo = line_center - window
    hi = line_center + window

    region = SpectralRegion(lo * u.AA, hi * u.AA)
    try:
        sub = extract_region(spectrum, region)
    except Exception:
        print(f"Warning: {lo}-{hi} Å not covered for this spectrum, skipping.")
        return None, None

    wave = sub.spectral_axis.value
    flux = sub.flux.value

    # fit continuum on shoulder regions only, masking out the line core
    shoulder_mask = (wave < line_center - shoulder) | (wave > line_center + shoulder)

    try:
        p_init = models.Chebyshev1D(1)
        fitter = fitting.LinearLSQFitter()
        p = fitter(p_init, wave[shoulder_mask], flux[shoulder_mask])
        continuum = p(wave)
        flux_norm = flux / continuum
    except Exception as e:
        print(f"Warning: continuum fit failed around {line_center} Å: {e}")
        return None, None

    return wave, flux_norm


# normalize around each feature
wave_ha,   flux_ha   = local_normalize(spectrum, line_center=6562.801, window=40, shoulder=8)   # H-alpha
wave_li,   flux_li   = local_normalize(spectrum, line_center=6707.76,  window=15, shoulder=4)   # Lithium — Fe line at 6707.4 nearby
wave_cahk, flux_cahk = local_normalize(spectrum, line_center=3950.0,   window=30, shoulder=8)   # Ca II H&K


# ==== 4-PANEL PLOT ====
figures, graphs = plt.subplots(2, 2, figsize=(10, 8))
plt.suptitle(f'\nLocally Normalized Key Spectral Features for {query}', fontsize=16)
plt.subplots_adjust(hspace=0.3)

# H-alpha
if wave_ha is not None:
    graphs[0,0].plot(wave_ha, flux_ha, 'tab:pink')
    graphs[0,0].axvline(x=6562.801, color='black', ls='--')
    graphs[0,0].axhline(y=1.0, color='gray', ls=':', linewidth=0.8)
    pad = 0.1 * (flux_ha.max() - flux_ha.min())
    graphs[0,0].set_ylim(flux_ha.min() - pad, flux_ha.max() + pad)
else:
    graphs[0,0].text(0.5, 0.5, 'Not covered', ha='center', va='center', transform=graphs[0,0].transAxes)
graphs[0,0].set_title(r'H-alpha 6562.801 $\AA$')
graphs[0,0].set_xlabel('Wavelength (Å)')
graphs[0,0].set_ylabel('Normalized Flux')

# Lithium + Fe line marker
if wave_li is not None:
    graphs[0,1].plot(wave_li, flux_li, 'tab:purple')
    graphs[0,1].axvline(x=6707.76, color='black', ls='--', label='Li 6707.76')
    graphs[0,1].axvline(x=6707.4,  color='red',   ls=':',  label='Fe 6707.4')
    graphs[0,1].axhline(y=1.0, color='gray', ls=':', linewidth=0.8)
    pad = 0.1 * (flux_li.max() - flux_li.min())
    graphs[0,1].set_ylim(flux_li.min() - pad, flux_li.max() + pad)
    graphs[0,1].legend(fontsize=7)
else:
    graphs[0,1].text(0.5, 0.5, 'Not covered', ha='center', va='center', transform=graphs[0,1].transAxes)
graphs[0,1].set_title(r'Lithium 6707.76 $\AA$')
graphs[0,1].set_xlabel('Wavelength (Å)')
graphs[0,1].set_ylabel('Normalized Flux')

# Ca II H
if wave_cahk is not None:
    graphs[1,0].plot(wave_cahk, flux_cahk, '#71C7A0')
    graphs[1,0].axvline(x=3933, color='black', ls='--')
    graphs[1,0].axhline(y=1.0, color='gray', ls=':', linewidth=0.8)
    graphs[1,0].set_xlim(3920, 3945)
    pad = 0.1 * (flux_cahk.max() - flux_cahk.min())
    graphs[1,0].set_ylim(flux_cahk.min() - pad, flux_cahk.max() + pad)
else:
    graphs[1,0].text(0.5, 0.5, 'Not covered', ha='center', va='center', transform=graphs[1,0].transAxes)
graphs[1,0].set_title(r'Ca II H 3933 $\AA$')
graphs[1,0].set_xlabel('Wavelength (Å)')
graphs[1,0].set_ylabel('Normalized Flux')

# Ca II K
if wave_cahk is not None:
    graphs[1,1].plot(wave_cahk, flux_cahk, '#4B9CD3')
    graphs[1,1].axvline(x=3968, color='black', ls='--')
    graphs[1,1].axhline(y=1.0, color='gray', ls=':', linewidth=0.8)
    graphs[1,1].set_xlim(3955, 3980)
    pad = 0.1 * (flux_cahk.max() - flux_cahk.min())
    graphs[1,1].set_ylim(flux_cahk.min() - pad, flux_cahk.max() + pad)
else:
    graphs[1,1].text(0.5, 0.5, 'Not covered', ha='center', va='center', transform=graphs[1,1].transAxes)
graphs[1,1].set_title(r'Ca II K 3968 $\AA$')
graphs[1,1].set_xlabel('Wavelength (Å)')
graphs[1,1].set_ylabel('Normalized Flux')

filename2 = f"Features_{query}_{instrument}_localnorm.png"
plt.savefig(filename2)
plt.show()