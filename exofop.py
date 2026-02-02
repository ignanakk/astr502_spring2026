# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 10:54:07 2026

@author: ilakk
"""

from exofop.download import System, SystemDownloader
import numpy as np

data_dir= "./tmp"
path= r"C:\Users\ilakk\OneDrive\Desktop\astr502\1_20_data"

system= System(name="GJ 1214")
system_loader= SystemDownloader(
    system=system,
    data_dir=path,)


tab = system_loader.spectroscopy.table
print(tab[:10])

tags= system_loader.spectroscopy.tags
sorted_tags = np.sort(tags)

system_loader.download(sorted_tags[:10],unzip=True)

system_loader.download(tags[:10], output_dir=None, unzip=True)  # type: ignore
print(f"Target directory: {system_loader.target_dir}")


system_loader.download_spectroscopy(output_dir=path, unzip=True)