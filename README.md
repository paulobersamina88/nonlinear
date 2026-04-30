# MDOF Pushover + RSA Reconciliation Streamlit App

This app helps reconcile STAAD-derived MDOF dynamic properties with a simplified nonlinear pushover workflow.

## Features
- Input STAAD-derived floor mass and lateral stiffness matrices
- Eigenvalue/modal analysis
- Modal response spectrum analysis using user-defined spectrum points
- First-mode pushover lateral force pattern
- Beam/column plastic moment capacity per frame and number of frames per axis
- Storey yield capacity estimation
- Nonlinear pushover curve generation
- ADRS capacity spectrum conversion and demand/capacity reconciliation

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy online
1. Upload `app.py`, `requirements.txt`, and this README to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app and select `app.py` as the main file.

## Engineering note
This is a teaching/reconciliation tool. It is not a substitute for full nonlinear static analysis in ETABS, SAP2000, SeismoStruct, SeismoBuild, OpenSees, or similar tools. The plastic mechanism and storey capacity equations are intentionally transparent so users can manually check them.
