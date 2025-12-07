# 🌌 CMB Power Spectrum Explorer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![CAMB](https://img.shields.io/badge/CAMB-1.5+-green.svg)](https://camb.info)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B.svg)](https://cmb-explorer-camb.streamlit.app/)

An interactive web application for exploring Cosmic Microwave Background (CMB) power spectra using CAMB with adjustable cosmological parameters. Built for astronomy research and education.

---

## 🚀 **[Try the Live App →](https://cmb-explorer-camb.streamlit.app/)**

**No installation required!** Experience the full-featured CMB explorer directly in your browser.

---

## 🎯 Motivation

The Cosmic Microwave Background provides a unique observational window into the early universe, merely 380,000 years after the Big Bang. Understanding how cosmological parameters shape the CMB power spectrum is essential for:

- 🔭 **Testing Inflationary Models** - Constraining the primordial power spectrum and tensor-to-scalar ratio
- 🧮 **Parameter Estimation** - Developing intuition for parameter degeneracies and constraints
- 📊 **Educational Research** - Bridging theoretical cosmology with hands-on exploration
- 🌌 **Preparation for Advanced Analysis** - Building skills for working with Planck, WMAP, and future CMB experiments

This project was developed to deepen practical understanding of CMB physics by enabling real-time exploration of how each cosmological parameter affects observable features—from acoustic peak heights to polarization patterns and the damping tail.

## ✨ Features

### 🔬 Cosmological Parameters (21 Total)
- **Primary ΛCDM:** H₀, Ωᵦh², Ωᶜh², τ, nₛ, Aₛ
- **Initial Power Spectrum:** Running of spectral index, pivot scales
- **CMB & BBN:** Temperature (2.7255 K), helium fraction
- **Extended Cosmology:** Neutrino masses, curvature, dark energy EoS, tensor-to-scalar ratio
- **Accuracy Controls:** Computation precision, lensing accuracy

### 📊 Visualizations
- Real-time CMB power spectra (TT, EE, BB, TE)
- Matter power spectrum P(k)
- Interactive Plotly plots with zoom/pan
- Log/linear scale toggle
- Professional astronomy-themed UI

### 💾 Data Management
- CSV export for power spectra
- TXT export for parameters
- Publication-quality output

### 🎓 Educational
- Parameter tooltips with physical interpretations
- 15+ scientific references (Planck 2018, CAMB papers)
- CMB physics explanations

## 🚀 Quick Start

### Option 1: Use the Live App (Recommended)

**🌐 [Launch App](https://cmb-explorer-camb.streamlit.app/)** - No installation needed!

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/Shivaji-137/CMB_Explorer.git
cd CMB_Explorer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run cmb_explorer_app.py
```

Or use the quick launch script:
```bash
chmod +x run_cmb_explorer.sh
./run_cmb_explorer.sh
```

### Requirements

```
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
camb>=1.5.0
plotly>=5.17.0
```

## 📖 Usage

1. **Adjust Parameters** - Use sliders in the sidebar
2. **Select Spectra** - Choose TT, EE, BB, or TE
3. **Click Compute** - Press "🚀 COMPUTE SPECTRA" button
4. **Explore Results** - View plots, derived parameters, export data

### Default Values (Planck 2018)
- H₀ = 67.32 km/s/Mpc
- Ωᵦh² = 0.0224
- Ωᶜh² = 0.1201
- τ = 0.0543
- nₛ = 0.9660
- Aₛ = 2.101 × 10⁻⁹

## 🧪 Testing

Verify installation:
```bash
python test_app.py
```
Expected: `5/5 tests passing`

## 🔧 Technical Details

- **Frontend:** Streamlit with custom CSS
- **Backend:** CAMB Python API
- **Visualization:** Plotly
- **Caching:** Optimized for performance
- **Stability:** Manual compute mode prevents crashes

### Performance
- Computation: 2-10 seconds (depends on lmax)
- Memory: ~300 MB
- Supports lmax up to 3000

## 📁 Project Structure

```
CMB_Explorer/
├── cmb_explorer_app.py      # Main Streamlit application (1,249 lines)
├── requirements.txt          # Python dependencies
├── run_cmb_explorer.sh       # Quick launch script
├── test_app.py              # Component verification (5 tests)
├── .streamlit/
│   └── config.toml          # UI theme configuration
├── README.md                # This file
├── LICENSE                  # MIT License
└── .gitignore              # Git ignore rules
```

### Key Components

- **Backend:** CAMB v1.5+ for CMB computation with 21 adjustable parameters
- **Frontend:** Streamlit with custom astronomy-themed CSS (dark space background, golden accents)
- **Visualization:** Plotly for interactive plots with zoom, pan, and hover details
- **Optimization:** Caching system prevents redundant computation, manual trigger prevents crashes
- **Export:** CSV for spectra data, TXT for parameter configurations

## 🎯 Use Cases

- **Research:** Parameter exploration, model validation, testing extended cosmologies
- **Education:** Teaching CMB physics, demonstrating parameter degeneracies
- **Analysis:** Reproduce Planck 2018 results, test inflationary predictions
- **Development:** Prototype for Bayesian parameter estimation pipelines

## 🌟 Key References

1. Planck Collaboration (2020). "Planck 2018 results. VI. Cosmological parameters." A&A 641, A6
2. Lewis, Challinor & Lasenby (2000). "Efficient computation of CMB anisotropies in closed FRW models." ApJ 538, 473
3. Lewis & Bridle (2002). "Cosmological parameters from CMB and other data: A Monte Carlo approach." PRD 66, 103511

Full references available in the app's built-in documentation.

## 🔮 Future Work

Building upon this foundation, planned extensions include:

### Phase 2: Advanced Analysis Tools
- **Bayesian Inference** - Implement MCMC/nested sampling for parameter estimation
- **Fisher Matrix Analysis** - Forecast parameter constraints for future experiments
- **Comparison Mode** - Test ΛCDM against alternative models (dynamical dark energy, modified gravity)
- **Data Integration** - Incorporate real Planck/WMAP data for model comparison

### Phase 3: Research Applications
- **Neutrino Mass Constraints** - Investigate CMB lensing implications for Σmν
- **Inflation Model Testing** - Discriminate between scenarios using primordial power spectrum running
- **Large-Scale Structure** - Integrate galaxy clustering data to break parameter degeneracies
- **21cm Cosmology** - Extend to high-redshift 21cm power spectra for reionization studies

### Phase 4: Community Features
- **Interactive Tutorials** - Guided parameter exploration exercises
- **Parameter Presets** - Quick access to WMAP, Planck 2015/2018, and theoretical models
- **Collaboration Tools** - Share parameter configurations and results
- **API Development** - Enable programmatic access for batch processing

These extensions will transform the tool from an educational explorer into a comprehensive research platform suitable for next-generation CMB experiments like Simons Observatory and CMB-S4.


## 📧 Contact

**Author:** Shivaji Chaulagain  
**Year:** 2025

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- CAMB development team (Antony Lewis et al.)
- Planck Collaboration
- Streamlit framework
- UI optimization: Claude AI & GitHub Copilot

---

**Built with ❤️ for Cosmology Research & Education**

*Version 2.0 | December 2025*
