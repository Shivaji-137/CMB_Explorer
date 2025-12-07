"""
Interactive CMB Power Spectrum Explorer
A production-ready Streamlit application for exploring CMB power spectra
using CAMB with adjustable cosmological parameters.
"""

import streamlit as st
import numpy as np
import pandas as pd
import camb
from camb import model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Page configuration
st.set_page_config(
    page_title="Interactive CMB Power Spectrum Explorer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for astronomy theme
st.markdown("""
<style>
    /* Main background with subtle star pattern */
    .stApp {
        background: linear-gradient(135deg, #0A0E27 0%, #1a1f3a 50%, #0f1535 100%);
    }
    
    /* Main header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFD700, #FFA500, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
    }
    
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #B8C5D6;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #0f1535 100%);
        border-right: 1px solid rgba(255, 215, 0, 0.2);
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFD700 !important;
        font-weight: 600;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #FFD700;
        font-weight: 700;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(26, 31, 58, 0.8);
        border: 1px solid rgba(255, 215, 0, 0.3);
        border-radius: 8px;
        color: #E0E0E0;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(26, 31, 58, 1);
        border-color: rgba(255, 215, 0, 0.5);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #000000 !important;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
        text-shadow: none !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #FFA500 0%, #FFD700 100%);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 215, 0, 0.4);
        color: #000000 !important;
    }
    
    .stButton>button p, .stButton>button div {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar button specific styling */
    [data-testid="stSidebar"] .stButton>button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #000000 !important;
        font-weight: 700;
        text-shadow: none !important;
    }
    
    [data-testid="stSidebar"] .stButton>button:hover {
        background: linear-gradient(135deg, #FFA500 0%, #FFD700 100%);
        color: #000000 !important;
    }
    
    [data-testid="stSidebar"] .stButton>button p {
        color: #000000 !important;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #357ABD 0%, #4A90E2 100%);
        box-shadow: 0 5px 15px rgba(74, 144, 226, 0.4);
    }
    
    /* DataFrame styling */
    .dataframe {
        background-color: rgba(26, 31, 58, 0.6) !important;
        border: 1px solid rgba(255, 215, 0, 0.2);
        border-radius: 8px;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: rgba(74, 144, 226, 0.2);
        border-left: 4px solid #4A90E2;
        color: #E0E0E0;
    }
    
    .stWarning {
        background-color: rgba(255, 165, 0, 0.2);
        border-left: 4px solid #FFA500;
        color: #E0E0E0;
    }
    
    .stSuccess {
        background-color: rgba(76, 175, 80, 0.2);
        border-left: 4px solid #4CAF50;
        color: #E0E0E0;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background-color: rgba(255, 215, 0, 0.2);
    }
    
    /* Text input and number input */
    input {
        background-color: rgba(26, 31, 58, 0.8) !important;
        border: 1px solid rgba(255, 215, 0, 0.3) !important;
        color: #E0E0E0 !important;
        border-radius: 5px;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: rgba(26, 31, 58, 0.8);
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
    
    /* Checkbox */
    .stCheckbox {
        color: #E0E0E0;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #FFD700 !important;
    }
    
    /* Subheaders */
    h2, h3 {
        color: #FFD700 !important;
        font-weight: 600;
    }
    
    /* Horizontal rule */
    hr {
        border-color: rgba(255, 215, 0, 0.3);
    }
    
    /* Footer styling */
    .footer-style {
        text-align: center;
        color: #B8C5D6;
        padding: 30px;
        font-family: 'Inter', sans-serif;
        background: rgba(26, 31, 58, 0.5);
        border-radius: 10px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)




def load_default_parameters():
    """Load default Planck 2018 cosmological parameters from planck_2018.ini"""
    defaults = {
        'H0': 67.32117,
        'ombh2': 0.0223828,
        'omch2': 0.1201075,
        'tau': 0.05430842,
        'ns': 0.9660499,
        'As': 2.100549e-9,
        'omnuh2': 0.0006451439,  # Neutrino density
        'mnu': 0.06,  # Sum of neutrino masses (eV)
        'omk': 0.0,  # Curvature
        'w': -1.0,  # Dark energy equation of state
    }
    return defaults


def compute_cmb_spectra(H0, ombh2, omch2, tau, ns, As, lmax=2500, 
                        mnu=0.06, omk=0.0, w=-1.0, r=0.0, 
                        accurate_lensing=False, nonlinear=0,
                        # Phase 1 additions
                        scalar_nrun=0.0, pivot_scalar=0.05,
                        temp_cmb=2.7255, helium_fraction=0.2454,
                        accuracy_boost=1.0, lens_pot_accuracy=1):
    """
    Compute CMB power spectra using CAMB.
    
    Parameters:
    -----------
    H0 : float - Hubble constant (km/s/Mpc)
    ombh2 : float - Baryon density parameter
    omch2 : float - Cold dark matter density parameter
    tau : float - Optical depth to reionization
    ns : float - Scalar spectral index
    As : float - Scalar amplitude (10^-9)
    lmax : int - Maximum multipole
    mnu : float - Sum of neutrino masses (eV)
    omk : float - Curvature parameter
    w : float - Dark energy equation of state
    r : float - Tensor-to-scalar ratio
    accurate_lensing : bool - Use higher accuracy for lensing
    nonlinear : int - Non-linear corrections (0=none, 1=matter, 2=CMB, 3=both)
    scalar_nrun : float - Running of spectral index
    pivot_scalar : float - Pivot scale in Mpc^-1
    temp_cmb : float - CMB temperature in K
    helium_fraction : float - Helium mass fraction
    accuracy_boost : float - Overall accuracy multiplier
    lens_pot_accuracy : int - Lensing potential accuracy (0-4)
    
    Returns:
    --------
    results : CAMB results object
    """
    try:
        # Set up CAMB parameters
        pars = camb.CAMBparams()
        
        # Set cosmological parameters (including neutrinos, curvature, and BBN)
        pars.set_cosmology(
            H0=H0,
            ombh2=ombh2,
            omch2=omch2,
            mnu=mnu,
            omk=omk,
            tau=tau,
            TCMB=temp_cmb,
            YHe=helium_fraction
        )
        
        # Dark energy equation of state
        pars.DarkEnergy.w = w
        
        # Set initial power spectrum parameters with running
        pars.InitPower.set_params(
            As=As,
            ns=ns,
            nrun=scalar_nrun,
            r=r,  # Tensor-to-scalar ratio
            pivot_scalar=pivot_scalar,
            pivot_tensor=pivot_scalar
        )
        
        # Set accuracy and output settings
        if accurate_lensing:
            lens_acc = max(2, lens_pot_accuracy)
        else:
            lens_acc = lens_pot_accuracy
        
        pars.set_for_lmax(lmax, lens_potential_accuracy=lens_acc)
        
        # Set accuracy boost
        pars.set_accuracy(
            AccuracyBoost=accuracy_boost,
            lAccuracyBoost=accuracy_boost
        )
        
        # Configure what to compute
        pars.WantTensors = (r > 0)  # Only compute tensors if r > 0
        pars.WantScalars = True
        pars.WantCls = True
        pars.WantTransfer = True  # Need this for sigma8 calculation
        
        # Lensing settings
        pars.DoLensing = True
        
        # Non-linear corrections
        # 0: linear, 1: non-linear matter, 2: non-linear CMB, 3: both
        if nonlinear == 1:
            pars.NonLinear = model.NonLinear_pk
        elif nonlinear == 2:
            pars.NonLinear = model.NonLinear_lens
        elif nonlinear == 3:
            pars.NonLinear = model.NonLinear_both
        else:
            pars.NonLinear = model.NonLinear_none
        
        # Set up for matter power spectrum (needed for sigma8)
        pars.set_matter_power(redshifts=[0.0], kmax=2.0)
        
        # Calculate results
        results = camb.get_results(pars)
        
        return results
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"❌ CAMB Error: {str(e)}")
        with st.expander("🔍 Click to see detailed error trace"):
            st.code(error_details, language="python")
        st.warning("""
        **Common issues:**
        - Check that parameters are in valid ranges
        - Ensure pivot_scalar > 0
        - Try lower accuracy_boost if computation fails
        - Verify temperature and helium fraction are reasonable
        """)
        return None


@st.cache_data(show_spinner=False)
def compute_and_extract_spectra(H0, ombh2, omch2, tau, ns, As, lmax=2500,
                                mnu=0.06, omk=0.0, w=-1.0, r=0.0,
                                accurate_lensing=False, nonlinear=0,
                                scalar_nrun=0.0, pivot_scalar=0.05,
                                temp_cmb=2.7255, helium_fraction=0.2454,
                                accuracy_boost=1.0, lens_pot_accuracy=1):
    """
    Compute CAMB spectra and extract data in one cached function.
    This prevents recomputation when only display options change.
    
    Returns:
    --------
    tuple : (data dict, derived_params dict)
    """
    try:
        # Compute CAMB results
        results = compute_cmb_spectra(
            H0, ombh2, omch2, tau, ns, As, lmax,
            mnu, omk, w, r, accurate_lensing, nonlinear,
            scalar_nrun, pivot_scalar, temp_cmb, helium_fraction,
            accuracy_boost, lens_pot_accuracy
        )
        
        if results is None:
            return None, None
        
        # Extract spectra
        powers = results.get_cmb_power_spectra(params=results.Params, CMB_unit='muK')
        totCL = powers['lensed_scalar']
        ell = np.arange(totCL.shape[0])
        
        data = {
            'ell': ell.tolist(),  # Convert to list for caching
            'TT': totCL[:, 0].tolist(),
            'EE': totCL[:, 1].tolist(),
            'BB': totCL[:, 2].tolist(),
            'TE': totCL[:, 3].tolist()
        }
        
        # Extract derived parameters
        omegam = (ombh2 + omch2) / (H0/100)**2
        omeganu = mnu / (93.14 * (H0/100)**2) if mnu > 0 else 0
        omegal = 1 - omegam - omeganu - omk
        age = results.get_derived_params()['age']
        
        try:
            sigma8 = results.get_sigma8_0()
        except:
            sigma8 = results.get_sigma8()[0]
        
        derived = {
            'omegam': omegam,
            'omeganu': omeganu,
            'omegal': omegal,
            'age': age,
            'sigma8': sigma8
        }
        
        return data, derived
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Error computing spectra: {str(e)}")
        with st.expander("🔍 Click to see detailed error"):
            st.code(error_details, language="python")
        return None, None


def extract_power_spectra(results, lmax=2500):
    """
    Extract power spectra from CAMB results.
    
    Returns:
    --------
    dict : Dictionary containing ell, TT, TE, EE, BB spectra
    """
    try:
        # Get power spectra (in μK^2)
        powers = results.get_cmb_power_spectra(params=results.Params, CMB_unit='muK')
        
        # Extract lensed spectra
        totCL = powers['lensed_scalar']
        
        # Multipole values
        ell = np.arange(totCL.shape[0])
        
        # Extract individual spectra (convert to D_ell = ell(ell+1)C_ell/(2π))
        # CAMB already returns D_ell for plotting
        data = {
            'ell': ell,
            'TT': totCL[:, 0],  # Temperature
            'EE': totCL[:, 1],  # E-mode polarization
            'BB': totCL[:, 2],  # B-mode polarization
            'TE': totCL[:, 3]   # Temperature-E-mode cross-correlation
        }
        
        return data
        
    except Exception as e:
        st.error(f"Error extracting spectra: {str(e)}")
        return None


def create_interactive_plot(data, selected_spectra, log_scale=False):
    """Create interactive Plotly plot of CMB power spectra with astronomy theme"""
    fig = go.Figure()
    
    colors = {
        'TT': '#00D9FF',
        'EE': '#FFD700',
        'BB': '#FF6B9D',
        'TE': '#7FFF00'
    }
    
    labels = {
        'TT': 'TT (Temperature)',
        'EE': 'EE (E-mode)',
        'BB': 'BB (B-mode)',
        'TE': 'TE (Temperature-E cross)'
    }
    
    ell = data['ell']
    
    for spectrum in selected_spectra:
        if spectrum in data:
            mask = ell >= 2
            fig.add_trace(go.Scatter(
                x=ell[mask],
                y=data[spectrum][mask],
                mode='lines',
                name=labels[spectrum],
                line=dict(color=colors[spectrum], width=2.5),
                hovertemplate='ℓ: %{x}<br>D<sub>ℓ</sub>: %{y:.2f} μK²<extra></extra>'
            ))
    
    y_axis_type = 'log' if log_scale else 'linear'
    
    fig.update_layout(
        title={'text': '<b>CMB Power Spectra</b>', 'font': {'size': 24, 'color': '#FFD700', 'family': 'Arial Black'}},
        xaxis={
            'title': {'text': '<b>Multipole moment ℓ</b>', 'font': {'size': 14, 'color': '#E0E0E0'}},
            'gridcolor': 'rgba(255, 215, 0, 0.15)',
            'showgrid': True,
            'zeroline': False,
            'color': '#E0E0E0'
        },
        yaxis={
            'title': {'text': '<b>D<sub>ℓ</sub> = ℓ(ℓ+1)C<sub>ℓ</sub>/(2π) [μK²]</b>', 'font': {'size': 14, 'color': '#E0E0E0'}},
            'type': y_axis_type,
            'gridcolor': 'rgba(255, 215, 0, 0.15)',
            'showgrid': True,
            'zeroline': False,
            'color': '#E0E0E0'
        },
        hovermode='x unified',
        plot_bgcolor='#0f1535',
        paper_bgcolor='#0f1535',
        height=600,
        legend={'x': 0.02, 'y': 0.98, 'bgcolor': 'rgba(26, 31, 58, 0.9)', 'bordercolor': 'rgba(255, 215, 0, 0.5)', 'borderwidth': 2, 'font': {'color': '#E0E0E0', 'size': 12}},
        margin={'l': 80, 'r': 40, 't': 80, 'b': 60}
    )
    
    return fig


def create_data_table(data, selected_spectra, max_ell=100):
    """Create a pandas DataFrame with spectra values"""
    df_dict = {'ell': data['ell'][:max_ell+1]}
    
    for spectrum in selected_spectra:
        if spectrum in data:
            df_dict[f'{spectrum} (μK²)'] = data[spectrum][:max_ell+1]
    
    df = pd.DataFrame(df_dict)
    return df


def parameter_tooltip(param_name):
    """Return educational tooltip for each parameter"""
    tooltips = {
        'H0': 'Hubble constant: Current expansion rate of the universe. Determines the age and size of the universe.',
        'ombh2': 'Baryon density: Density of ordinary matter (protons, neutrons). Affects acoustic oscillation amplitudes.',
        'omch2': 'Cold dark matter density: Non-baryonic matter density. Influences structure formation and peak heights.',
        'tau': 'Optical depth: Related to reionization epoch. Suppresses small-scale temperature fluctuations.',
        'ns': 'Spectral index: Tilt of primordial power spectrum. ns<1 means more power on large scales.',
        'As': 'Scalar amplitude: Overall amplitude of primordial fluctuations. Sets CMB temperature fluctuation scale.'
    }
    return tooltips.get(param_name, '')


def main():
    """Main application"""
    
    # Header with astronomy design
    st.markdown('<p class="main-header">🌌 CMB Power Spectrum Explorer 🔭</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">✨ Interactive Cosmological Parameter Analysis • Explore the Early Universe ✨</p>', unsafe_allow_html=True)
    
    # Add astronomy-themed info banner
    st.markdown("""
    <div style='background: linear-gradient(90deg, rgba(26, 31, 58, 0.8) 0%, rgba(15, 21, 53, 0.8) 100%); 
                padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 215, 0, 0.3); 
                margin-bottom: 20px; text-align: center;'>
        <span style='font-size: 1.1rem; color: #FFD700; font-weight: 600;'>
            🪐 Analyze the Cosmic Microwave Background radiation from 380,000 years after the Big Bang 🌠
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for parameters with astronomy styling
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px; background: linear-gradient(135deg, #1a1f3a, #0f1535); 
                border-radius: 10px; margin-bottom: 15px; border: 2px solid rgba(255, 215, 0, 0.4);'>
        <h2 style='color: #FFD700; margin: 0; font-size: 1.5rem;'>🔬 Parameters</h2>
        <p style='color: #B8C5D6; font-size: 0.9rem; margin: 5px 0 0 0; font-style: italic;'>
            Adjust to explore the cosmos
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load defaults
    defaults = load_default_parameters()
    
    # Parameter inputs with tooltips
    st.sidebar.subheader("Primary Parameters")
    
    H0 = st.sidebar.slider(
        "H₀ (km/s/Mpc)",
        min_value=60.0,
        max_value=80.0,
        value=defaults['H0'],
        step=0.5,
        help=parameter_tooltip('H0')
    )
    
    ombh2 = st.sidebar.slider(
        "Ωᵦh² (Baryon density)",
        min_value=0.015,
        max_value=0.030,
        value=defaults['ombh2'],
        step=0.0001,
        format="%.4f",
        help=parameter_tooltip('ombh2')
    )
    
    omch2 = st.sidebar.slider(
        "Ωᶜh² (CDM density)",
        min_value=0.08,
        max_value=0.16,
        value=defaults['omch2'],
        step=0.001,
        format="%.4f",
        help=parameter_tooltip('omch2')
    )
    
    tau = st.sidebar.slider(
        "τ (Optical depth)",
        min_value=0.01,
        max_value=0.15,
        value=defaults['tau'],
        step=0.005,
        format="%.3f",
        help=parameter_tooltip('tau')
    )
    
    ns = st.sidebar.slider(
        "nₛ (Spectral index)",
        min_value=0.90,
        max_value=1.05,
        value=defaults['ns'],
        step=0.005,
        format="%.4f",
        help=parameter_tooltip('ns')
    )
    
    As = st.sidebar.slider(
        "Aₛ × 10⁹ (Scalar amplitude)",
        min_value=1.5,
        max_value=3.0,
        value=defaults['As'] * 1e9,
        step=0.05,
        format="%.2f",
        help=parameter_tooltip('As')
    ) * 1e-9
    
    # Display options
    st.sidebar.subheader("Display Options")
    
    selected_spectra = st.sidebar.multiselect(
        "Select Spectra to Display",
        options=['TT', 'TE', 'EE', 'BB'],
        default=['TT', 'EE'],
        help="Choose which power spectra to plot"
    )
    
    log_scale = st.sidebar.checkbox(
        "Logarithmic Y-axis",
        value=False,
        help="Use logarithmic scale for better visibility of all features"
    )
    
    lmax = st.sidebar.slider(
        "Maximum ℓ",
        min_value=500,
        max_value=3000,
        value=2500,
        step=100,
        help="Maximum multipole moment to compute"
    )
    
    # Advanced parameters section
    st.sidebar.subheader("Advanced Parameters")
    
    # Phase 1: Initial Power Spectrum parameters
    with st.sidebar.expander("🔬 Initial Power Spectrum"):
        scalar_nrun = st.slider(
            "dns/dlnk (Running of ns)",
            min_value=-0.1,
            max_value=0.1,
            value=0.0,
            step=0.001,
            format="%.3f",
            help="Running of spectral index. dns/dlnk quantifies scale-dependence of ns. Planck: ~-0.003 (consistent with 0)."
        )
        
        pivot_scalar = st.number_input(
            "Pivot Scale k₀ (Mpc⁻¹)",
            min_value=0.01,
            max_value=0.1,
            value=0.05,
            step=0.01,
            format="%.3f",
            help="Pivot scale where ns and As are defined. Planck uses 0.05 Mpc⁻¹."
        )
    
    # Phase 1: CMB Temperature & Helium
    with st.sidebar.expander("🌡️ CMB & BBN"):
        temp_cmb = st.number_input(
            "CMB Temperature T₀ (K)",
            min_value=2.7,
            max_value=2.8,
            value=2.7255,
            step=0.0001,
            format="%.4f",
            help="Current CMB temperature. COBE/FIRAS: 2.7255 ± 0.0006 K."
        )
        
        helium_fraction = st.slider(
            "Helium Fraction Yₚ",
            min_value=0.20,
            max_value=0.30,
            value=0.2454,
            step=0.0001,
            format="%.4f",
            help="Primordial helium mass fraction from BBN. Planck 2018: 0.2454."
        )
    
    # Phase 1: Accuracy Settings
    with st.sidebar.expander("🎯 Accuracy Settings"):
        accuracy_boost = st.slider(
            "Accuracy Boost",
            min_value=1.0,
            max_value=3.0,
            value=1.0,
            step=0.5,
            format="%.1f",
            help="Overall accuracy multiplier. >1 increases precision but slows computation. Use 2-3 for publication-quality results."
        )
        
        lens_pot_accuracy = st.slider(
            "Lensing Accuracy",
            min_value=0,
            max_value=4,
            value=1,
            step=1,
            help="Lensing potential accuracy for B-modes. 0=fastest, 1=default, 2-4=high accuracy. Use 2+ for ℓ>2000 or precise BB."
        )
    
    with st.sidebar.expander("🔬 Advanced Settings"):
        mnu = st.slider(
            "Σmᵥ (Neutrino masses, eV)",
            min_value=0.0,
            max_value=0.5,
            value=defaults.get('mnu', 0.06),
            step=0.01,
            format="%.2f",
            help="Sum of neutrino masses. Planck: ~0.06 eV. Affects small-scale damping."
        )
        
        omk = st.slider(
            "Ωₖ (Curvature)",
            min_value=-0.02,
            max_value=0.02,
            value=defaults.get('omk', 0.0),
            step=0.001,
            format="%.3f",
            help="Spatial curvature. Ωₖ=0 for flat universe (Planck constraint)."
        )
        
        w = st.slider(
            "w (Dark energy EoS)",
            min_value=-1.5,
            max_value=-0.5,
            value=defaults.get('w', -1.0),
            step=0.01,
            format="%.2f",
            help="Dark energy equation of state. w=-1 is cosmological constant (ΛCDM)."
        )
        
        r = st.slider(
            "r (Tensor-to-scalar ratio)",
            min_value=0.0,
            max_value=0.2,
            value=0.0,
            step=0.01,
            format="%.2f",
            help="Primordial gravitational waves. r=0 means no tensors. Upper limit: r<0.06."
        )
        
        accurate_lensing = st.checkbox(
            "High-accuracy lensing",
            value=False,
            help="Use lens_potential_accuracy=2 for better B-mode accuracy (slower)"
        )
        
        nonlinear = st.selectbox(
            "Non-linear corrections",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "None (linear)",
                1: "Matter power (HALOFIT)",
                2: "CMB lensing (HALOFIT)",
                3: "Both matter + CMB"
            }[x],
            index=0,
            help="Include non-linear effects. Planck 2018 used option 3."
        )
    
    # Reset button
    if st.sidebar.button("Reset to Planck 2018 Values"):
        st.rerun()
    
    # Add spacing
    st.sidebar.markdown("---")
    
    # COMPUTE BUTTON - Manual trigger
    st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); 
                padding: 5px; border-radius: 8px; text-align: center; margin: 10px 0;'>
        <p style='color: #0A0E27; font-weight: 700; font-size: 0.9rem; margin: 5px;'>
            ⚠️ Adjust all parameters first, then click below
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    compute_button = st.sidebar.button(
        "🚀 COMPUTE SPECTRA",
        type="primary",
        use_container_width=True,
        help="Click to compute CMB spectra with current parameters. Prevents crashes while adjusting sliders."
    )
    
    # Main content
    if not selected_spectra:
        st.warning("⚠️ Please select at least one spectrum to display.")
        return
    
    # Check if compute button was clicked or if we have cached results
    if not compute_button and 'last_computed_params' not in st.session_state:
        st.info("👆 **Adjust parameters in the sidebar, then click 'COMPUTE SPECTRA' to run.**")
        st.markdown("""
        <div style='background: rgba(255, 215, 0, 0.1); padding: 20px; border-radius: 10px; 
                    border: 2px solid rgba(255, 215, 0, 0.3); margin: 20px 0;'>
            <h3 style='color: #FFD700; text-align: center;'>🎯 How to Use</h3>
            <ol style='color: #E0E0E0; line-height: 2;'>
                <li>Adjust all cosmological parameters in the sidebar</li>
                <li>Select which spectra to display (TT, EE, BB, TE)</li>
                <li>Choose display options (log scale, max ℓ)</li>
                <li>Click <strong style='color: #FFD700;'>🚀 COMPUTE SPECTRA</strong> button</li>
                <li>Wait for computation to complete</li>
                <li>View and download results!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Store current parameters
    current_params = (H0, ombh2, omch2, tau, ns, As, lmax, mnu, omk, w, r,
                     accurate_lensing, nonlinear, scalar_nrun, pivot_scalar,
                     temp_cmb, helium_fraction, accuracy_boost, lens_pot_accuracy)
    
    # Only compute if button clicked OR parameters match cached
    if compute_button:
        st.session_state['last_computed_params'] = current_params
    elif 'last_computed_params' in st.session_state:
        # Check if parameters changed
        if current_params != st.session_state['last_computed_params']:
            st.warning("⚠️ Parameters changed. Click **COMPUTE SPECTRA** to update results.")
            return
    
    # Show current parameters being used
    with st.expander("🔍 Current Parameters (click to verify)"):
        st.write(f"""
        **Primary:** H0={H0}, ωb={ombh2}, ωc={omch2}, τ={tau}, ns={ns}, As={As:.2e}  
        **Power Spectrum:** nrun={scalar_nrun}, k₀={pivot_scalar}  
        **BBN:** T_CMB={temp_cmb}K, Yp={helium_fraction}  
        **Accuracy:** boost={accuracy_boost}, lens_pot={lens_pot_accuracy}  
        **Advanced:** Σmν={mnu}, Ωk={omk}, w={w}, r={r}  
        **Computation:** lmax={lmax}, lensing={accurate_lensing}, nonlinear={nonlinear}
        """)
    
    # Compute spectra (cached - won't recompute on display option changes)
    with st.spinner('🔭 Computing CMB power spectra... This may take a few seconds.'):
        try:
            data, derived = compute_and_extract_spectra(
                H0, ombh2, omch2, tau, ns, As, lmax,
                mnu, omk, w, r,
                accurate_lensing, nonlinear,
                scalar_nrun, pivot_scalar,
                temp_cmb, helium_fraction,
                accuracy_boost, lens_pot_accuracy
            )
            
            if data is None or derived is None:
                st.error("❌ Failed to compute spectra. See error details above.")
                st.info("💡 Try using default Planck 2018 values (Reset button) or check parameter ranges.")
                return
        except Exception as e:
            st.error(f"❌ Computation failed: {str(e)}")
            st.info("💡 Click 'Reset to Planck 2018 Values' to use known-good parameters.")
            return
        
        # Convert lists back to numpy arrays for plotting
        data = {
            'ell': np.array(data['ell']),
            'TT': np.array(data['TT']),
            'EE': np.array(data['EE']),
            'BB': np.array(data['BB']),
            'TE': np.array(data['TE'])
        }
    
    # Display plot
    st.markdown("""
    <div style='text-align: center; margin: 30px 0 20px 0;'>
        <h2 style='color: #FFD700; font-size: 2rem; font-weight: 700;'>
            📊 Power Spectra Visualization 🌟
        </h2>
    </div>
    """, unsafe_allow_html=True)
    fig = create_interactive_plot(data, selected_spectra, log_scale)
    st.plotly_chart(fig, width='stretch')
    
    # Use cached derived parameters
    omegam = derived['omegam']
    omeganu = derived['omeganu']
    omegal = derived['omegal']
    age = derived['age']
    sigma8 = derived['sigma8']
    
    # Display parameters in expanders to save space
    with st.expander("🔭 Current Parameter Values", expanded=False):
        params_df = pd.DataFrame({
            'Parameter': ['H₀', 'Ωᵦh²', 'Ωᶜh²', 'τ', 'nₛ', 'Aₛ × 10⁹'],
            'Value': [f'{H0:.2f}', f'{ombh2:.4f}', f'{omch2:.4f}', 
                     f'{tau:.4f}', f'{ns:.4f}', f'{As*1e9:.3f}'],
            'Unit': ['km/s/Mpc', '', '', '', '', '']
        })
        st.dataframe(params_df, hide_index=True, width='stretch')
    
    with st.expander("🌟 Derived Parameters", expanded=False):
        derived_df = pd.DataFrame({
            'Parameter': ['Ωₘ', 'Ωᵥ', 'ΩΛ', 'Ωₖ', 'Age', 'σ₈'],
            'Value': [f'{omegam:.4f}', f'{omeganu:.5f}', f'{omegal:.4f}', 
                     f'{omk:.4f}', f'{age:.3f}', f'{sigma8:.4f}'],
            'Unit': ['', '', '', '', 'Gyr', '']
        })
        st.dataframe(derived_df, hide_index=True, width='stretch')
    
    # Data table section
    with st.expander("📈 View Numerical Data Table (first 100 multipoles)"):
        df = create_data_table(data, selected_spectra, max_ell=100)
        st.dataframe(df, width='stretch')
    
    # Download section - in expander to save space
    with st.expander("💾 Export Data", expanded=False):
        # Prepare full CSV
        full_df = create_data_table(data, selected_spectra, max_ell=lmax)
        csv = full_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Spectra as CSV",
            data=csv,
            file_name=f"cmb_spectra_H0_{H0:.1f}_ns_{ns:.3f}.csv",
            mime="text/csv",
            help="Download the computed power spectra data",
            width='stretch'
        )
        
        # Prepare parameters file
        params_text = f"""CMB Power Spectra Parameters
Generated by Interactive CMB Power Spectrum Explorer

Cosmological Parameters:
H0 = {H0:.4f} km/s/Mpc
ombh2 = {ombh2:.6f}
omch2 = {omch2:.6f}
tau = {tau:.6f}
ns = {ns:.6f}
As = {As:.6e}
mnu = {mnu:.3f} eV
omk = {omk:.6f}
w = {w:.3f}
r = {r:.3f}

Initial Power Spectrum:
scalar_nrun = {scalar_nrun:.6f}
pivot_scalar = {pivot_scalar:.3f} Mpc^-1

CMB & BBN:
temp_cmb = {temp_cmb:.4f} K
helium_fraction = {helium_fraction:.4f}

Derived Parameters:
Omega_m = {omegam:.6f}
Omega_nu = {omeganu:.6f}
Omega_Lambda = {omegal:.6f}
Omega_k = {omk:.6f}
Age = {age:.4f} Gyr
sigma_8 = {sigma8:.4f}

Computation Settings:
lmax = {lmax}
accurate_lensing = {accurate_lensing}
nonlinear = {nonlinear}
accuracy_boost = {accuracy_boost:.1f}
lens_potential_accuracy = {lens_pot_accuracy}
"""
        
        st.download_button(
            label="📥 Download Parameters as TXT",
            data=params_text,
            file_name=f"cmb_parameters_H0_{H0:.1f}_ns_{ns:.3f}.txt",
            mime="text/plain",
            help="Download the parameter values used",
            width='stretch'
        )
    
    # Matter power spectrum (optional)
    if st.checkbox("🌌 Show Matter Power Spectrum", value=False):
        st.markdown("""
        <div style='text-align: center; margin: 30px 0 20px 0;'>
            <h2 style='color: #FFD700; font-size: 1.8rem; font-weight: 700;'>
                🌠 Matter Power Spectrum 🔬
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Computing matter power spectrum..."):
            try:
                # Need to recompute results for matter power spectrum
                # This is only done if the checkbox is enabled
                results_for_pk = compute_cmb_spectra(
                    H0, ombh2, omch2, tau, ns, As, lmax,
                    mnu, omk, w, r,
                    accurate_lensing, nonlinear,
                    scalar_nrun, pivot_scalar,
                    temp_cmb, helium_fraction,
                    accuracy_boost, lens_pot_accuracy
                )
                
                if results_for_pk is None:
                    raise Exception("Could not compute CAMB results")
                
                # Get matter power spectrum (already computed)
                # Use maxkh=2 to match the kmax set in compute_cmb_spectra
                # This prevents segmentation faults from requesting k values beyond computed range
                kh, z_pk, pk = results_for_pk.get_matter_power_spectrum(
                    minkh=1e-4, maxkh=2.0, npoints=200
                )
                
                # Plot with astronomy theme
                fig_pk = go.Figure()
                colors_pk = ['#00D9FF', '#FFD700', '#FF6B9D']
                
                for i, (zi, color) in enumerate(zip(z_pk, colors_pk)):
                    fig_pk.add_trace(go.Scatter(
                        x=kh,
                        y=pk[i, :],
                        mode='lines',
                        name=f'z = {zi:.1f}',
                        line=dict(color=color, width=2.5)
                    ))
                
                fig_pk.update_layout(
                    title={'text': '<b>Matter Power Spectrum P(k)</b>', 'font': {'size': 22, 'color': '#FFD700', 'family': 'Arial Black'}},
                    xaxis={
                        'title': {'text': '<b>k/h (Mpc⁻¹)</b>', 'font': {'size': 14, 'color': '#E0E0E0'}},
                        'type': 'log',
                        'gridcolor': 'rgba(255, 215, 0, 0.15)',
                        'color': '#E0E0E0'
                    },
                    yaxis={
                        'title': {'text': '<b>P(k) [(Mpc/h)³]</b>', 'font': {'size': 14, 'color': '#E0E0E0'}},
                        'type': 'log',
                        'gridcolor': 'rgba(255, 215, 0, 0.15)',
                        'color': '#E0E0E0'
                    },
                    hovermode='x unified',
                    plot_bgcolor='#0f1535',
                    paper_bgcolor='#0f1535',
                    height=500,
                    legend={'x': 0.02, 'y': 0.98, 'bgcolor': 'rgba(26, 31, 58, 0.9)', 'bordercolor': 'rgba(255, 215, 0, 0.5)', 'borderwidth': 2, 'font': {'color': '#E0E0E0', 'size': 12}}
                )
                
                st.plotly_chart(fig_pk, width='stretch')
                
                st.info(f"σ₈(z=0) = {sigma8:.4f} - RMS matter fluctuations in 8 Mpc/h spheres")
                
            except Exception as e:
                st.error(f"Could not compute matter power spectrum: {str(e)}")
                st.warning("Try adjusting cosmological parameters or disable matter power spectrum display.")
    
    # Educational information
    with st.expander("🌌 About the CMB Power Spectra"):
        st.markdown("""
        <div style='color: #E0E0E0; line-height: 1.8;'>
        
        <h3 style='color: #FFD700; font-size: 1.5rem;'>🔬 Understanding CMB Power Spectra</h3>
        
        <p style='color: #B8C5D6; font-size: 1.05rem;'>
        The Cosmic Microwave Background (CMB) power spectra show how temperature and polarization 
        fluctuations vary across different angular scales in the universe.
        </p>
        
        <h4 style='color: #FFD700; margin-top: 20px;'>📡 Spectrum Types:</h4>
        <ul style='color: #E0E0E0;'>
            <li><span style='color: #00D9FF; font-weight: 700;'>TT (Temperature-Temperature)</span>: Temperature fluctuations. Shows acoustic oscillations from 
            the early universe. The peaks correspond to different modes of oscillation.</li>
            <li><span style='color: #FFD700; font-weight: 700;'>EE (E-mode Polarization)</span>: Gradient-type polarization pattern. Generated by Thomson scattering.</li>
            <li><span style='color: #FF6B9D; font-weight: 700;'>BB (B-mode Polarization)</span>: Curl-type polarization. At large scales, can be generated by 
            primordial gravitational waves.</li>
            <li><span style='color: #7FFF00; font-weight: 700;'>TE (Temperature-E-mode Cross)</span>: Correlation between temperature and E-mode polarization.</li>
        </ul>
        
        <h4 style='color: #FFD700; margin-top: 20px;'>🎯 Key Features:</h4>
        <ul style='color: #E0E0E0;'>
            <li><span style='font-weight: 700;'>First Peak</span> (ℓ ≈ 220): Related to the physical size of the sound horizon at recombination.</li>
            <li><span style='font-weight: 700;'>Second/Third Peaks</span>: Compression and rarefaction modes.</li>
            <li><span style='font-weight: 700;'>Damping Tail</span> (ℓ > 1000): Small-scale fluctuations damped by photon diffusion.</li>
        </ul>
        
        <h4 style='color: #FFD700; margin-top: 20px;'>⚙️ Parameter Effects:</h4>
        <ul style='color: #E0E0E0;'>
            <li>Higher <span style='color: #FFD700;'>H₀</span> shifts peaks to higher ℓ (smaller angular scales).</li>
            <li>Higher <span style='color: #FFD700;'>Ωᵦh²</span> increases odd peak heights relative to even peaks.</li>
            <li>Higher <span style='color: #FFD700;'>Ωᶜh²</span> increases all peak heights.</li>
            <li>Higher <span style='color: #FFD700;'>τ</span> suppresses small-scale power.</li>
            <li><span style='color: #FFD700;'>nₛ</span> < 1 means more power on large scales (smaller ℓ).</li>
            <li><span style='color: #FFD700;'>dns/dlnk</span> (running): Scale-dependent tilt. Constrains inflation models.</li>
            <li>Lower <span style='color: #FFD700;'>T_CMB</span> or <span style='color: #FFD700;'>Yₚ</span>: Affects recombination epoch and peak structure.</li>
        </ul>
        
        <h4 style='color: #FFD700; margin-top: 20px;'>🔬 Advanced Features:</h4>
        <ul style='color: #E0E0E0;'>
            <li><span style='font-weight: 700;'>Running of ns</span>: Measures scale-dependence of primordial spectrum. Tests inflation models.</li>
            <li><span style='font-weight: 700;'>Pivot Scale</span>: Reference k₀ where As and ns are defined. Standard: 0.05 Mpc⁻¹.</li>
            <li><span style='font-weight: 700;'>Accuracy Boost</span>: Increases integration precision. Use >1 for publication-quality.</li>
            <li><span style='font-weight: 700;'>Lensing Accuracy</span>: Critical for B-modes. Use 2-4 for ℓ>2000 or precise gravitational wave constraints.</li>
        </ul>
        
        </div>
        """, unsafe_allow_html=True)
    
    # References section
    with st.expander("📚 References & Citations"):
        # Header
        st.markdown("<h3 style='color: #FFD700; font-size: 1.5rem;'>📖 Key References</h3>", unsafe_allow_html=True)
        
        # CAMB Documentation
        st.markdown("<h4 style='color: #FFD700; margin-top: 20px;'>CAMB Documentation & Code:</h4>", unsafe_allow_html=True)
        st.markdown("""
        - **CAMB Documentation:** [https://camb.readthedocs.io/](https://camb.readthedocs.io/)
        - **CAMB Website:** [https://camb.info](https://camb.info)
        - **CAMB Notes (PDF):** [Lewis (2025)](https://cosmologist.info/notes/CAMB.pdf)
        """)
        
        # Primary CAMB Papers
        st.markdown("<h4 style='color: #FFD700; margin-top: 20px;'>Primary CAMB Papers:</h4>", unsafe_allow_html=True)
        st.markdown("""
        - **Lewis, Challinor & Lasenby (2000)** - "Efficient computation of CMB anisotropies in closed FRW models"  
          *ApJ 538, 473-476* | [arXiv:astro-ph/9911177](https://arxiv.org/abs/astro-ph/9911177)
        
        - **Howlett, Lewis, Hall & Challinor (2012)** - "CMB power spectrum parameter degeneracies in the era of precision cosmology"  
          *JCAP 04, 027* | [arXiv:1201.3654](https://arxiv.org/abs/1201.3654)
        """)
        
        # CosmoMC & Analysis Tools
        st.markdown("<h4 style='color: #FFD700; margin-top: 20px;'>CosmoMC & Analysis Tools:</h4>", unsafe_allow_html=True)
        st.markdown("""
        - **Lewis & Bridle (2002)** - "Cosmological parameters from CMB and other data: A Monte Carlo approach"  
          *Phys. Rev. D 66, 103511* | [arXiv:astro-ph/0205436](https://arxiv.org/abs/astro-ph/0205436)
        
        - **Lewis (2013)** - "Efficient sampling of fast and slow cosmological parameters"  
          *Phys. Rev. D 87, 103529* | [arXiv:1304.4473](https://arxiv.org/abs/1304.4473)
        
        - **Lewis (2019)** - "GetDist: a Python package for analysing Monte Carlo samples"  
          *JCAP 08, 025 (2025)* | [arXiv:1910.13970](https://arxiv.org/abs/1910.13970)
        
        - **Torrado & Lewis (2021)** - "Cobaya: Code for Bayesian Analysis of hierarchical physical models"  
          *JCAP 05, 057* | [arXiv:2005.05290](https://arxiv.org/abs/2005.05290)
        """)
        
        # Planck Mission Results
        st.markdown("<h4 style='color: #FFD700; margin-top: 20px;'>Planck Mission Results:</h4>", unsafe_allow_html=True)
        st.markdown("""
        - **Planck Collaboration (2020)** - "Planck 2018 results. VI. Cosmological parameters"  
          *A&A 641, A6* | [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)
        
        - **Planck Collaboration (2020)** - "Planck 2018 results. VIII. Gravitational lensing"  
          *A&A 641, A8* | [arXiv:1807.06210](https://arxiv.org/abs/1807.06210)
        """)
        
        # Theoretical Background
        st.markdown("<h4 style='color: #FFD700; margin-top: 20px;'>Theoretical Background:</h4>", unsafe_allow_html=True)
        st.markdown("""
        - **Seljak & Zaldarriaga (1996)** - "A Line of sight integration approach to cosmic microwave background anisotropies"  
          *ApJ 469, 437-444* | [arXiv:astro-ph/9603033](https://arxiv.org/abs/astro-ph/9603033)  
          *(CMBFAST - precursor to CAMB)*
        
        - **Zaldarriaga, Seljak & Bertschinger (1998)** - "Integral solution for the microwave background anisotropies in nonflat universes"  
          *ApJ 494, 491-502* | [arXiv:astro-ph/9704265](https://arxiv.org/abs/astro-ph/9704265)
        """)
        
        # CAMB Sources (Advanced Features)
        st.markdown("<h4 style='color: #FFD700; margin-top: 20px;'>CAMB Sources (Advanced Features):</h4>", unsafe_allow_html=True)
        st.markdown("""
        - **Challinor & Lewis (2011)** - "The linear power spectrum of observed source number counts"  
          *Phys. Rev. D 84, 043516* | [arXiv:1105.5292](https://arxiv.org/abs/1105.5292)
        
        - **Lewis & Challinor (2007)** - "The 21cm angular-power spectrum from the dark ages"  
          *Phys. Rev. D 76, 083005* | [arXiv:astro-ph/0702600](https://arxiv.org/abs/astro-ph/0702600)
        """)
        
        # How to Cite
        st.markdown("<h4 style='color: #FFD700; margin-top: 20px;'>📝 How to Cite This Tool:</h4>", unsafe_allow_html=True)
        st.info("""
        **If using this tool in your research, please cite:**
        
        - **CAMB:** Lewis, Challinor & Lasenby (2000) and Howlett et al. (2012)
        - **Planck Data:** Planck Collaboration (2020)
        - **This Tool:** Chaulagain, S. (2025), CMB Explorer v2.0 [Software]
        """)
        
        # Additional Resources
        st.markdown("<h4 style='color: #FFD700; margin-top: 20px;'>🔗 Additional Resources:</h4>", unsafe_allow_html=True)
        st.markdown("""
        - **Cosmologist.info:** [https://cosmologist.info](https://cosmologist.info)
        - **Planck Legacy Archive:** [https://pla.esac.esa.int](https://pla.esac.esa.int)
        - **arXiv Cosmology:** [astro-ph.CO](https://arxiv.org/list/astro-ph.CO/recent)
        """)
    
    # Footer with astronomy styling
    st.markdown("---")
    st.markdown("""
    <div class='footer-style'>
        <p style='font-size: 1.5rem; font-weight: 700; margin-bottom: 10px; color: #FFD700;'>
            🌌 CMB Explorer v2.0 🔭
        </p>
        <p style='font-size: 1rem; opacity: 0.9; color: #B8C5D6; margin-bottom: 8px;'>
            Built with ❤️ for Cosmology • Powered by CAMB v{}
        </p>
        <p style='font-size: 0.95rem; opacity: 0.85; margin-top: 8px; color: #A8B8D0;'>
            💻 Code written by <span style='color: #FFD700; font-weight: 600;'>Shivaji Chaulagain</span>
        </p>
        <p style='font-size: 0.9rem; opacity: 0.8; margin-top: 5px; color: #9BACC8;'>
            🤖 UI built and optimization with <span style='color: #00D9FF;'>Claude AI</span> using <span style='color: #7FFF00;'>GitHub Copilot</span>
        </p>
        <p style='font-size: 0.85rem; opacity: 0.7; margin-top: 10px; color: #9BACC8;'>
            🌟 Research & Educational Tool for Astronomy 🌟
        </p>
    </div>
    """.format(camb.__version__), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
