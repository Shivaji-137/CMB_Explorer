#!/usr/bin/env python3
"""
Test script to verify the CMB Explorer app components work correctly
Run this before launching the full Streamlit app
"""

import sys

def test_imports():
    """Test that all required packages are importable"""
    print("Testing imports...")
    try:
        import streamlit as st
        print("  ✓ streamlit:", st.__version__)
    except ImportError as e:
        print(f"  ✗ streamlit: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✓ numpy:", np.__version__)
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import pandas as pd
        print("  ✓ pandas:", pd.__version__)
    except ImportError as e:
        print(f"  ✗ pandas: {e}")
        return False
    
    try:
        import camb
        print("  ✓ camb:", camb.__version__)
    except ImportError as e:
        print(f"  ✗ camb: {e}")
        return False
    
    try:
        import plotly
        print("  ✓ plotly:", plotly.__version__)
    except ImportError as e:
        print(f"  ✗ plotly: {e}")
        return False
    
    return True


def test_camb_computation():
    """Test that CAMB can compute spectra"""
    print("\nTesting CAMB computation...")
    try:
        import camb
        
        # Set up basic parameters
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.32, ombh2=0.022, omch2=0.12)
        pars.InitPower.set_params(As=2.1e-9, ns=0.965)
        pars.set_for_lmax(100, lens_potential_accuracy=1)
        
        # Try to compute
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(params=results.Params, CMB_unit='muK')
        
        if powers is not None and 'lensed_scalar' in powers:
            print("  ✓ CAMB computation successful")
            print(f"    Computed spectra up to ℓ={powers['lensed_scalar'].shape[0]-1}")
            return True
        else:
            print("  ✗ CAMB computation failed: No output")
            return False
            
    except Exception as e:
        print(f"  ✗ CAMB computation failed: {e}")
        return False


def test_app_syntax():
    """Test that the main app file has no syntax errors"""
    print("\nTesting app syntax...")
    try:
        import py_compile
        py_compile.compile('cmb_explorer_app.py', doraise=True)
        print("  ✓ App syntax is valid")
        return True
    except py_compile.PyCompileError as e:
        print(f"  ✗ App syntax error: {e}")
        return False


def test_file_structure():
    """Test that all necessary files exist"""
    print("\nTesting file structure...")
    import os
    
    required_files = [
        'cmb_explorer_app.py',
        'requirements.txt',
        'README_CMB_Explorer.md',
        'QUICK_START.md',
        'run_cmb_explorer.sh'
    ]
    
    all_exist = True
    for filename in required_files:
        if os.path.exists(filename):
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} not found")
            all_exist = False
    
    return all_exist


def test_plotly():
    """Test basic plotly functionality"""
    print("\nTesting Plotly...")
    try:
        import plotly.graph_objects as go
        import numpy as np
        
        # Create a simple test plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
        
        # Check if figure has data
        if len(fig.data) > 0:
            print("  ✓ Plotly figure creation successful")
            return True
        else:
            print("  ✗ Plotly figure creation failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Plotly test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("CMB Power Spectrum Explorer - Component Test")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure", test_file_structure),
        ("App Syntax", test_app_syntax),
        ("Plotly Functionality", test_plotly),
        ("CAMB Computation", test_camb_computation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The app is ready to run.")
        print("\nTo start the app, run:")
        print("  ./run_cmb_explorer.sh")
        print("or")
        print("  streamlit run cmb_explorer_app.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("You may need to install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
