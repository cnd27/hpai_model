# Simulation model for highly pathogenic avian influenza (HPAI)


Included python files are for MCMC fitting and model simulation for a between poultry premises transmission model for highly pathogenic avian influenza (HPAI).

---

## Python files

run_hpai_model.py runs the model to generate the figures seen in the manuscript.

hpai_model.py is the model code for fitting and simulations.


---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/cnd27/hpai_model.git
cd hpai_model
```

### 2. Set up dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
cd src
python3 run_hpai_model.py 
```

---

## Files in this repo

```
HPAI_control_measures/
├── data/                         # Input data
├── output/                       # Output files
├── src/                          # Python scripts
│   └── run_hpai_model.py         # Plotting script
│   └── model.py                  # Main model functions
├── requirements.txt              # Dependencies
└── README.md                     # Instructions

```

---

## Requirements

See `requirements.txt` for full list and exact versions, but the key libraries used are:

- `geopandas`
- `matplotlib`
- `numpy`
- `pandas`
- `scipy` 

---

## Author

Christopher Davis, University of Warwick

GitHub: [@cnd27](https://github.com/cnd27)

---

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

