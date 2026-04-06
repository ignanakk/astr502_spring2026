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
from astropy import constants as const
import time


# 1. QUERY
eso = Eso()
# eso.login('YOUR_USERNAME') 

query = 'HATS-34'
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

'''
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

plt.show()
'''

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
print('Teff Values:')
print(Teff_columns)

Teff_mean = np.mean(simbad_result['mesfe_h.teff']) #average Teff measurement
print(f'Mean Teff: {Teff_mean}')

#RV correction
#rv_kms = 36.2119305876328  #from SIMBAD database
rv_kms= rv

c_kms = const.c.to('km/s').value
xvals_rest = xvals / (1 + rv_kms / c_kms)

spectrum2 = Spectrum(
    spectral_axis=xvals_rest * u.AA,
    flux=yvals * u.Unit('erg / (cm^2 s AA)')
)


# normalize around each feature
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

#equivalent width calculations
from specutils.analysis import equivalent_width
'''
# H-alpha EW
if wave_ha2 is not None:#checks that wavelengths region is covered by spectrum
    spectrum_ha = Spectrum(
        spectral_axis=wave_ha2 * u.AA, #u.AA attaches units (Angstroms)
        flux=flux_ha2 * u.dimensionless_unscaled   #no units (after normalization, flux is just a ratio)
    )
    ew_ha = equivalent_width(spectrum_ha, continuum=1, regions=SpectralRegion(6555 * u.AA, 6570 * u.AA)) #equivalent width calculations
    print(f"H-alpha EW: {ew_ha:.3f}")
'''
# Lithium EW
if wave_li2 is not None:
    spectrum_li = Spectrum(
        spectral_axis=wave_li2 * u.AA, 
        flux=flux_li2 * u.dimensionless_unscaled
    )
    ew_li = equivalent_width(spectrum_li, continuum=1, regions=SpectralRegion(6704 * u.AA, 6711 * u.AA))
    print(f"Lithium EW #1: {ew_li:.3f}")
'''
# Ca II H
if wave_cahk2 is not None:
    spectrum_cah = Spectrum(
        spectral_axis=wave_cahk2 * u.AA,
        flux=flux_cahk2 * u.dimensionless_unscaled
    )
    ew_cah = equivalent_width(spectrum_cah, continuum=1, regions=SpectralRegion(3929 * u.AA, 3937 * u.AA))
    print(f"Ca II H EW: {ew_cah:.3f}")

# Ca II K
if wave_cahk2 is not None:
    spectrum_cak = Spectrum(
        spectral_axis=wave_cahk2 * u.AA,
        flux=flux_cahk2 * u.dimensionless_unscaled
    )
    ew_cak = equivalent_width(spectrum_cak, continuum=1, regions=SpectralRegion(3964 * u.AA, 3972 * u.AA))
    print(f"Ca II K EW: {ew_cak:.3f}")
'''
from specutils.analysis import equivalent_width
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

#with help of CLAUDE to help figure out what was wrong
def fit_gaussian_center(wave, flux, center_guess, search_window=1.5):
    mask = (wave >= center_guess - search_window) & (wave <= center_guess + search_window)
    w, f = wave[mask], flux[mask]

    if len(w) < 5:
        print(f"  Too few points near {center_guess:.2f} Å to fit — using guess.")
        return center_guess

    # Invert so the absorption dip becomes a peak for fitting
    f_inv = 1.0 - f

    def gaussian(x, amp, cen, sigma, offset):
        return amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2) + offset

    try:
        p0 = [f_inv.max(), center_guess, 0.3, 0.0]
        bounds = (
            [0,      center_guess - search_window, 0.01, -0.5],
            [2.0,    center_guess + search_window, 2.0,   0.5],
        )
        popt, _ = curve_fit(gaussian, w, f_inv, p0=p0, bounds=bounds, maxfev=5000)
        fitted_center = popt[1]
        print(f"  Gaussian fit center: {fitted_center:.4f} Å  (guess was {center_guess:.4f} Å)")
        return fitted_center
    except Exception as e:
        print(f"  Gaussian fit failed ({e}), using guess {center_guess:.4f} Å")
        return center_guess


def compute_ew(wave, flux, nominal_center, search_window=0.2, ew_half_width=0.2, label="line"):
    
    if wave is None or flux is None:
        print(f"  {label}: spectrum region not available.")
        return None

    # Detect center
    center = fit_gaussian_center(wave, flux, nominal_center, search_window)

    # Build Spectrum object for just the integration window
    lo = (center - ew_half_width) * u.AA
    hi = (center + ew_half_width) * u.AA

    # Clip to available wavelength range
    if lo.value < wave.min() or hi.value > wave.max():
        print(f"  {label}: integration window [{lo:.2f}, {hi:.2f}] partly outside spectrum — clipping.")
        lo = max(lo.value, wave.min() + 0.01) * u.AA
        hi = min(hi.value, wave.max() - 0.01) * u.AA

    sp = Spectrum(
        spectral_axis=wave * u.AA,
        flux=flux * u.dimensionless_unscaled
    )

    try:
        ew = equivalent_width(sp, continuum=1, regions=SpectralRegion(lo, hi))
        # specutils returns negative for absorption — flip sign
        ew_mA = -ew.to(u.AA).value * 1000  # convert Å → mÅ
        print(f"  {label} EW #2 = {ew_mA:.3f} mÅ  (center: {center:.4f} Å)")
        return ew_mA
    except Exception as e:
        print(f"  {label} EW calculation failed: {e}")
        return None


# ── Run EW measurements ───────────────────────────────────────────────────────
print("\n=== Equivalent Width Measurements ===")
'''

# H-alpha — wide window is fine here, line is strong
ew_ha = compute_ew(
    wave_ha2, flux_ha2,
    nominal_center=6562.801,
    search_window=2.0,
    ew_half_width=3.0,
    label="H-alpha"
)
'''

# Lithium — narrow window; Fe I at 6707.4 Å
# Nominal rest wavelength for Li: 6707.76 Å
ew_li = compute_ew(
    wave_li2, flux_li2,
    nominal_center=6707.76,
    search_window=0.5,   
    #ew_half_width=0.36,   
    label="Lithium 6707.76"
)

'''
# Ca II H (3933.66 Å)
ew_cah = compute_ew(
    wave_cahk2, flux_cahk2,
    nominal_center=3933.66,
    search_window=2.0,
    ew_half_width=2.0,
    label="Ca II H"
)

# Ca II K (3968.47 Å)
ew_cak = compute_ew(
    wave_cahk2, flux_cahk2,
    nominal_center=3968.47,
    search_window=2.0,
    ew_half_width=2.0,
    label="Ca II K"
)
'''

# ── Diagnostic plot: show exactly what was integrated for Li ─────────────────
if wave_li2 is not None and ew_li is not None:
    li_center = fit_gaussian_center(wave_li2, flux_li2, 6707.76, search_window=1.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wave_li2, flux_li2, 'tab:purple', label='Normalized flux')
    ax.axvline(li_center,  color='black', ls='--', label=f'Li fit center {li_center:.3f} Å')
    ax.axvline(6707.4,     color='red',   ls=':',  label='Fe I 6707.4 Å')
    ax.axhline(1.0,        color='gray',  ls=':',  lw=0.8)
    ax.axvspan(li_center - .35, li_center + .35, alpha=0.15, color='purple', label='EW integration window')
    ax.set_title(f'Lithium 6707.76 Å — EW = {ew_li:.1f} mÅ')
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Normalized Flux')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    
#import CSV
import pandas as pd
df = pd.read_csv(r'C:\Users\ilakk\OneDrive\Desktop\astr502\astr502_spring2026\ASTR502_Master_Parameters_List.csv')

df_query = df.loc[df['hostname'] == query]
t_eff = df_query['tic_teff'].item()
print(f't_eff: {t_eff}')


def ew_diagnostic(wavelength, flux, li_center=6707.8, window=6.0):
    
    mask = (wavelength > li_center - window/2) & (wavelength < li_center + window/2)
    wl = wavelength[mask]
    fl = flux[mask]

    edge = (wl < li_center - window/3) | (wl > li_center + window/3)
    continuum = np.median(fl[edge])

    ew_val = np.sum((1 - fl)) * np.mean(np.diff(wl)) * 1000
    
    # ── plot ──
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(f'Li EW Diagnostic — EW = {ew_val:.1f} mÅ', fontsize=13)

    # Panel 1: raw flux + continuum + zones
    ax1 = axes[0]
    ax1.plot(wl, fl, 'k-', lw=1.5, label='Flux')
    ax1.axhline(continuum, color='r', ls='--', lw=1.5, label=f'Continuum = {continuum:.4f}')
    ax1.plot(wl[edge], fl[edge], 'r.', ms=7, zorder=5, label='Continuum points (edges)')
    ax1.axvspan(li_center - window/2, li_center - window/3,
                alpha=0.15, color='red',  label='Edge zone (continuum)')
    ax1.axvspan(li_center + window/3, li_center + window/2,
                alpha=0.15, color='red')
    ax1.axvspan(li_center - window/3, li_center + window/3,
                alpha=0.15, color='blue', label='Integration zone')
    ax1.axvline(li_center, color='blue', ls=':', lw=1)
    ax1.set_ylabel('Flux')
    ax1.legend(fontsize=8)
    ax1.set_title('Raw flux — red dots = continuum estimate points')

    # Panel 2: normalised flux + shaded EW area
    fl_norm = fl / continuum
    ax2 = axes[1]
    ax2.plot(wl, fl_norm, 'k-', lw=1.5, label='Normalised flux')
    ax2.axhline(1.0, color='r', ls='--', lw=1, label='Continuum = 1')
    ax2.axvline(li_center, color='blue', ls=':', lw=1, label='Li 6707.8 Å')

    # shade only the integration zone
    int_zone = (wl > li_center - window/3) & (wl < li_center + window/3)
    ax2.fill_between(wl[int_zone], fl_norm[int_zone], 1.0,
                     alpha=0.35, color='blue', label=f'EW area = {ew_val:.1f} mÅ')

    ax2.axvspan(li_center - window/2, li_center - window/3,
                alpha=0.15, color='red', label='Edge zone')
    ax2.axvspan(li_center + window/3, li_center + window/2,
                alpha=0.15, color='red')
    ax2.set_ylabel('Normalised Flux')
    ax2.set_xlabel('Wavelength (Å)')
    ax2.legend(fontsize=8)
    ax2.set_title('Normalised — blue shading is what gets integrated')

    plt.tight_layout()
    plt.show()

    return ew_val


li_ew = ew_diagnostic(wave_li2, flux_li2, li_center=6707.76, window=6.0)
print(f'Lithium EW: {li_ew:.3f} mÅ')



from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

result = NasaExoplanetArchive.query_object('HATS-34', table='pscomppars')
vsini=result['st_vsin'][0].value # stellar vsini in km/s

c= 3e5 # km/s
li_center = 6707.76

broadening_ang = (vsini / c) * li_center
window = max(2.0, 6 * broadening_ang)  # 3σ each side, minimum 2 Å
li_ew = ew_diagnostic(wave_li2, flux_li2, li_center=6707.76, window=window)

print(f'Lithium EW #3: {li_ew:.3f} mÅ')

li_ew = ew_diagnostic(wave_li2, flux_li2, li_center=6707.76, window=6)
