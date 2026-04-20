# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:29:36 2026

@author: ilakk
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:46:48 2026

@author: ilakk
"""


import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from astroquery.eso import Eso
from astropy.io import fits
from astropy import constants as const
import time


# 1. QUERY
eso = Eso()
eso.login(username='ignanakk', store_password=True)

query = 'K2-210'

instrument = 'HARPS' 
results = eso.query_surveys(target=query, surveys=instrument)
print(results)

plot_dir = rf'C:\Users\ilakk\OneDrive\Desktop\astr502\Plots\{query}'
os.makedirs(plot_dir, exist_ok=True)

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
plt.title(f"Full Spectrum for {query}")
plt.xlabel("Wavelength (Å)")
plt.ylabel("Flux")
plt.show()
'''

# Detailed Feature Subplots
fig, graphs = plt.subplots(2, 2, figsize=(12, 10))
plt.suptitle(f'Key Spectral Features: {query}', fontsize=16)

def plot_feature(ax, wave, flux, center, window, title, color):
    # Define the zoom window
    mask = (wave >= center - window) & (wave <= center + window)
    
    if not np.any(mask):
        ax.text(0.5, 0.5, f'Feature {center}Å\nNot in Range', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
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
'''
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

# ==== LITHIUM PLOT ====
fig, ax = plt.subplots(figsize=(8, 5))
plt.suptitle(f'\nLocally Normalized Key Spectral Features for {query}', fontsize=8)

# Lithium + Fe line marker
if wave_li is not None:
    ax.plot(wave_li, flux_li, 'tab:purple')
    ax.axvline(x=6707.76, color='black', ls='--', label='Li 6707.76')
    ax.axvline(x=6707.4,  color='red',   ls=':',  label='Fe 6707.4')
    ax.axhline(y=1.0, color='gray', ls=':', linewidth=0.8)
    pad = 0.1 * (flux_li.max() - flux_li.min())
    ax.set_ylim(flux_li.min() - pad, flux_li.max() + pad)
    ax.legend(fontsize=7)
else:
    ax.text(0.5, 0.5, 'Not covered', ha='center', va='center', transform=ax.transAxes)

ax.set_title(r'Lithium 6707.76 $\AA$')
ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel('Normalized Flux')
plt.show()

#RV value from SIMBAD
from astroquery.simbad import Simbad

all_fields=Simbad.list_votable_fields() #see what column names are
print(all_fields)

from astroquery.simbad import Simbad

Simbad.reset_votable_fields()
Simbad.add_votable_fields('rvz_radvel')
Simbad.add_votable_fields('mesfe_h')

simbad_result = Simbad.query_object(query)

print(simbad_result.colnames) #check column names

rv   = simbad_result['rvz_radvel'][0]
Teff_columns = simbad_result['mesfe_h.teff'] #multiple measurements
fe_h = simbad_result['mesfe_h.fe_h'] #metallicity

print(rv)

#RV correction
#rv_kms = 36.2119305876328  #from SIMBAD database
rv_kms= rv

c_kms = const.c.to('km/s').value
xvals_rest = xvals / (1 + rv_kms / c_kms)

# First-pass spectrum with SIMBAD RV only — used to detect residual shift
spectrum_rv1 = Spectrum(
    spectral_axis=xvals_rest * u.AA,
    flux=yvals * u.Unit('erg / (cm^2 s AA)')
)

wave_li_rv1, flux_li_rv1 = local_normalize(spectrum_rv1, line_center=6707.76, window=15, shoulder=4)

# Detect residual shift from first-pass Li line
if wave_li_rv1 is not None:
    search_mask_rv1 = (wave_li_rv1 > 6706.5) & (wave_li_rv1 < 6709.0)
    detected_li_center = wave_li_rv1[search_mask_rv1][np.argmin(flux_li_rv1[search_mask_rv1])]
    residual_shift = detected_li_center - 6707.76
    print(f"Detected Li center: {detected_li_center:.3f} Å  |  Residual shift after RV correction: {residual_shift:+.3f} Å")
else:
    detected_li_center = 6707.76
    residual_shift = 0.0
    print("Warning: could not detect Li line for residual correction, using 0.")

# Apply residual correction on top of SIMBAD RV correction
xvals_rest_corrected = xvals_rest + residual_shift

# Final spectrum with full correction applied
spectrum2 = Spectrum(
    spectral_axis=xvals_rest_corrected * u.AA,
    flux=yvals * u.Unit('erg / (cm^2 s AA)')
)

# normalize around each feature using fully corrected spectrum
wave_ha2,   flux_ha2   = local_normalize(spectrum2, line_center=6562.801, window=40, shoulder=8)   # H-alpha
wave_li2,   flux_li2   = local_normalize(spectrum2, line_center=6707.76,  window=15, shoulder=4)   # Lithium — Fe line at 6707.4 nearby
wave_cahk2, flux_cahk2 = local_normalize(spectrum2, line_center=3950.0,   window=30, shoulder=8)   # Ca II H&K


# ==== 4-PANEL PLOT ====
figures, graphs = plt.subplots(2, 2, figsize=(10, 8))
plt.suptitle(f'\nLocally Normalized + RV Corrected Key Spectral Features for {query}', fontsize=16)
plt.subplots_adjust(hspace=0.3)

# H-alpha
if wave_ha is not None:
    graphs[0,0].plot(wave_ha2, flux_ha2, 'tab:pink')
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
    graphs[0,1].plot(wave_li2, flux_li2, 'tab:purple')
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
    graphs[1,0].plot(wave_cahk2, flux_cahk2, '#71C7A0')
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
    graphs[1,1].plot(wave_cahk2, flux_cahk2, '#4B9CD3')
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
plt.show()

# ==== LITHIUM PLOT with empirical center detection ====
fig, ax = plt.subplots(figsize=(8, 5))

if wave_li2 is not None:
    # Find the actual line minimum near the expected Li position (should now be ~0 after full correction)
    search_mask = (wave_li2 > 6706.5) & (wave_li2 < 6709.0)
    detected_li_center = wave_li2[search_mask][np.argmin(flux_li2[search_mask])]
    residual_shift2 = detected_li_center - 6707.76
    print(f"Post-correction Li center: {detected_li_center:.3f} Å  |  Remaining residual: {residual_shift2:+.3f} Å")

    ax.plot(wave_li2, flux_li2, 'tab:purple')
    ax.axvline(x=detected_li_center, color='black', ls='--',
               label=f'Li detected: {detected_li_center:.3f} Å')
    ax.axvline(x=6707.76, color='black', ls=':', alpha=0.4,
               label='Li rest: 6707.76 Å')
    ax.axvline(x=6707.4 + residual_shift2, color='red', ls=':',
               label=f'Fe I (shifted): {6707.4 + residual_shift2:.3f} Å')
    ax.axhline(y=1.0, color='gray', ls=':', linewidth=0.8)
    pad = 0.1 * (flux_li2.max() - flux_li2.min())
    ax.set_ylim(flux_li2.min() - pad, flux_li2.max() + pad)
    ax.legend(fontsize=7)
else:
    ax.text(0.5, 0.5, 'Not covered', ha='center', va='center', transform=ax.transAxes)

ax.set_title(f'Lithium 6707.76 Å — {query}\n'
             f'Detected center: {detected_li_center:.3f} Å  (residual offset: {residual_shift2:+.3f} Å)')
plt.suptitle('\nNormalized + Residual Correction', fontsize=8)

ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel('Normalized Flux')
plt.tight_layout()
plt.show()

def ew_diagnostic(wavelength, flux, li_center=6707.8, window=6.0, search_window=1.5,
                  ew_half_width=0.5):
    mask = (wavelength > li_center - window/2) & (wavelength < li_center + window/2)
    wl = wavelength[mask]
    fl = flux[mask]

    # Detect actual line minimum
    search_mask = (wl > li_center - search_window) & (wl < li_center + search_window)
    if np.any(search_mask):
        detected_center = wl[search_mask][np.argmin(fl[search_mask])]
        print(f"  Nominal: {li_center:.3f} Å | Detected minimum: {detected_center:.3f} Å "
              f"(offset: {detected_center - li_center:+.3f} Å)")
    else:
        detected_center = li_center
        print(f"  Warning: using nominal center {li_center:.3f} Å")

    # Continuum: use only the far edges of the window, avoiding ALL line features
    edge = (
        ((wl < detected_center - window/2.5) | (wl > detected_center + window/2.5))
    )
    # Further exclude any flux more than 1.5% below the 90th percentile (absorption features)
    flux_90 = np.percentile(fl[edge], 90)
    edge = edge & (fl > flux_90 * 0.985)

    if np.sum(edge) < 4:
        print("  Warning: too few continuum points, relaxing threshold.")
        edge = (wl < detected_center - window/3) | (wl > detected_center + window/3)

    p = np.polyfit(wl[edge], fl[edge], 1)
    continuum = np.polyval(p, wl)

    # EW: integrate where normalized flux is actually below continuum
    fl_norm_temp = fl / continuum
    
    # Find where flux rises back above threshold on each side of detected center
    threshold = 0.995
    blue_side = wl[wl < detected_center]
    red_side  = wl[wl > detected_center]
    fl_blue   = fl_norm_temp[wl < detected_center]
    fl_red    = fl_norm_temp[wl > detected_center]
    
    # Walk outward from center until flux exceeds threshold
    above_blue = blue_side[fl_blue > threshold]
    above_red  = red_side[fl_red > threshold]
    
    blue_lim = above_blue[-1] if len(above_blue) > 0 else detected_center - ew_half_width
    red_lim  = above_red[0]   if len(above_red)  > 0 else detected_center + ew_half_width
    
    int_mask = (wl >= blue_lim) & (wl <= red_lim)
    wl_int   = wl[int_mask]
    fl_int   = fl[int_mask]
    cont_int = continuum[int_mask]

    if len(wl_int) < 3:
        print("  Too few points in integration window.")
        return None, detected_center

    integrand = 1 - fl_int / cont_int
    ew_val = np.trapz(integrand, wl_int) * 1000  # mÅ

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    fig.suptitle(f'{query} Li EW Diagnostic — EW = {ew_val:.1f} mÅ\n'
                 f'Detected center: {detected_center:.3f} Å  (nominal: {li_center:.3f} Å)', fontsize=12)
    LI_REST = 6707.76
    ax1 = axes[0]
    ax1.plot(wl, fl, 'k-', lw=1.5, label='Flux')
    ax1.plot(wl, continuum, 'r--', lw=1.5, label='Linear continuum fit')
    ax1.plot(wl[edge], fl[edge], 'r.', ms=7, zorder=5, label='Continuum anchor points')
    ax1.axvline(detected_center, color='blue', ls='--', lw=1.2, label=f'Detected: {detected_center:.3f} Å')
    ax1.axvline(LI_REST,       color='gray', ls=':',  lw=1.0, label=f'Li rest: {LI_REST:.2f} Å')
    ax1.axvspan(detected_center - ew_half_width, detected_center + ew_half_width,
                alpha=0.15, color='blue', label='EW integration window')
    ax1.set_ylabel('Flux')
    ax1.legend(fontsize=8)
    ax1.set_title('Raw flux with linear continuum')

    fl_norm = fl / continuum
    ax2 = axes[1]
    ax2.plot(wl, fl_norm, 'k-', lw=1.5, label='Normalised flux')
    ax2.axhline(1.0, color='r', ls='--', lw=1)
    ax2.axvline(detected_center, color='blue', ls='--', lw=1.2, label=f'Detected: {detected_center:.3f} Å')
    ax2.axvline(LI_REST,       color='gray', ls=':',  lw=1.0, label=f'Li rest: {LI_REST:.2f} Å')
    ax2.axvline(6707.4,          color='red',  ls=':',  lw=1.0, label='Fe I 6707.4 Å')
    ax2.fill_between(wl_int, fl_int / cont_int, 1.0,
                     alpha=0.35, color='blue', label=f'EW = {ew_val:.1f} mÅ')
    ax2.set_ylabel('Normalised Flux')
    ax2.set_xlabel('Wavelength (Å)')
    ax2.legend(fontsize=8)
    ax2.set_title('Normalised — blue = integrated area')

    plt.tight_layout()
    plt.show()
    print(f'lithium ew: {ew_val}')
    return ew_val, detected_center

li_ew, li_detected = ew_diagnostic(wave_li2, flux_li2, 
                                   li_center=detected_li_center,
                                   window=6.0, 
                                   search_window=0.3,
                                   ew_half_width=1.5)  # wider to catch full trough

