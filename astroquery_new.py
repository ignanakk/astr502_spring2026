# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 10:40:33 2026

@author: ilakk
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 11:34:41 2026

@author: ilakk
"""

from astroquery.eso import Eso
from astropy.io import fits
import numpy as np
import os
import requests

eso = Eso()

query = "WASP_136"
results = eso.query_surveys(target=query, survey='HARPS')

#Google Gemini AI:
if len(results) > 0:
    # Print columns so you can see what's available
    print(f"Available columns: {results.colnames}")
    
    # Try to find the ID. We'll check 'DP_ID' first, then 'ARCFILE', 
    # then just take the first column if those aren't found.
    if 'DP_ID' in results.colnames:
        file_id = results['DP_ID'][0]
    elif 'ARCFILE' in results.colnames:
        file_id = results['ARCFILE'][0]
    else:
        file_id = results[0][0] # Fallback to the very first cell
        
    print(f"Targeting File ID: {file_id}")
    
    path = r"C:\Users\ilakk\OneDrive\Desktop\astr502"
    if not os.path.exists(path):
        os.makedirs(path)
        
    # Clean the filename for Windows
    clean_filename = str(file_id).replace(":", "-") + ".fits"
    full_save_path = os.path.join(path, clean_filename)

    # Use the download URL
    url = f"https://dataportal.eso.org/dataPortal/file/{file_id}"
    print(f"Downloading to: {full_save_path}...")
    
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(full_save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        print("Download complete!")
        
        # Verify the FITS file
        try:
            with fits.open(full_save_path) as hdul:
                hdul.info()
                print(f"\nObject in Header: {hdul[0].header.get('OBJECT')}")
        except Exception as e:
            print(f"Downloaded, but couldn't read FITS: {e}")
            
    else:
        print(f"Download failed. Status: {response.status_code}")
        print("Note: If this is 401, you must run eso.login() first.")
else:
    print("No results found.")
    
    
# below code generated with help of ChatGPT to get only spectral results
# if the column exists
if "Product category" in results.colnames:
    spec_rows = [("SPECTRUM" in str(x)) for x in results["Product category"]]
    spec = results[spec_rows]
else:
    spec = results
print("Total rows:", len(results))
print("Spectrum-like rows:", len(spec))
print(spec[:10])

# below code generated with help of ChatGPT to get only files containing spectral info
# choose which table to download from: spec if non-empty, else results
use = spec if len(spec) > 0 else results
# pick first row for now
arc = use["ARCFILE"][0]
print("Downloading ARCFILE:", arc)
downloaded = eso.retrieve_data(arc)
print("Downloaded:", downloaded)

# below code generated with help of ChatGPT to check if file type downloaded correctly as BinTableHDU
path = downloaded[0] if isinstance(downloaded, (list, tuple)) else downloaded
hdul = fits.open(path)
hdul.info()
# WAVE / FLUX INFO IN BinTableHDU 