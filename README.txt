MTACD - Multi-Temporal Aerial Change Detection
Helsinki Urban Change Detection
Programming 2 project - Arcada UAS 2026
================================================

What does this do?
------------------
The app downloads aerial photos of Helsinki from the city's open map service,
trains a neural network to find buildings in them, and then compares two photos
from different years to see what changed (new buildings, demolished buildings etc).

It has a GUI with 4 tabs. You basically go left to right through them:

  Generate Data - downloads tiles from Helsinki's WMS API and creates
  building masks from the WFS building footprint data. This is your training data.

  Train Model - trains the U-Net (or Siamese U-Net) on whatever data you
  generated. Can also use your own images if you have them. Saves the model
  with a timestamp so you don't overwrite old ones by accident.

  Download Pairs - type a place name like "Kalasatama" or "Pasila", pick a
  before and after year, and it downloads + normalises both images. Helsinki
  has aerial photos going back to the 1930s which is pretty cool.

  Detect Changes - load a before/after pair, run the model, get a coloured
  output showing what appeared (green) or was demolished (red) between the photos.


How to run it
-------------
1. Run Install.bat first (only need to do this once, downloads ~2GB of packages)
2. Run Run.bat to open the app
3. Generate some tiles (I'd recommend at least 200-300 for decent results)
4. Train the model - 60 epochs takes a while but gives better results
5. Download a pair for somewhere in Helsinki and run detection

Needs Python 3.12 specifically because the AMD GPU library (torch-directml)
doesn't support 3.13 yet. If you only have 3.13 it will just run on CPU which
is slower but works fine.


Files
-----
app.py          - the main GUI app
pipeline.py     - the U-Net model and all the training/inference code
batch_train.py  - for running training overnight without the GUI (see RunBatch.bat)
Install.bat     - installs dependencies
Run.bat         - launches the app
requirements.txt

models/         - trained model checkpoints go here
data/           - training tiles and masks
pairs/          - downloaded image pairs


Where the data comes from
--------------------------
Aerial images come from the City of Helsinki's open WMS service (kartta.hel.fi).
Building outlines come from the WFS service on the same server - it gives back
GeoJSON polygons which get rasterised into masks.
Geocoding uses Nominatim (OpenStreetMap), no API key needed.

All free and open data, no accounts required.


Known issues / things that might go wrong
------------------------------------------
- If the model predicts almost nothing (black output), try lowering the threshold
  slider in the Detect tab. Or train on more data.

- "Location not found" in the pairs tab - add ", Helsinki" to the place name,
  Nominatim sometimes needs the city to narrow it down.

- Detection on CPU is slow for large images, can take 20-30 seconds. AMD GPU
  speeds this up a lot via DirectML.

- The model sometimes picks up roads or large flat roofs as buildings. Generating
  more varied tiles (different parts of the city) helps with this.

- Results are not perfect - this is a limitation of training on automatically
  generated masks from the WFS data rather than hand-labelled data.
