# Reproducibility Notes

This repository includes the key CSV, figure, and report outputs used in the submitted analysis.

## Recommended Environment

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- Packages listed in `requirements.txt`

## Run Order

```bash
pip install -r requirements.txt
jupyter notebook notebooks/비정형네트워크프로젝트.ipynb
```

The notebook can request live web data from Leaguepedia/Fandom, OP.GG, GOL.GG, and Riot Data Dragon. If any website changes its HTML/API behavior or blocks requests, live collection may fail. The included CSV files are provided so the analysis can still be reviewed.

## Important Interpretation Rules

- The counter network is a match-result-based matchup advantage network, not a pure 1:1 lane counter table.
- OP champion status uses OP.GG patch-tier labels as an external definition.
- Prediction results should be treated as supporting evidence. In this run, adding network features did not improve model performance.

