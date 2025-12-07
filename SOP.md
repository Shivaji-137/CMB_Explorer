# Research Statement: CMB Power Spectrum Explorer

## Project Motivation and Research Context

My fascination with the Cosmic Microwave Background (CMB) stems from its unique position as a direct observational window into the early universe, merely 380,000 years after the Big Bang. The CMB power spectrum encodes fundamental information about cosmological parameters that govern our universe's evolution, structure formation, and ultimate fate. To deepen my understanding of how these parameters shape observable CMB features, I developed an interactive web-based application called the **CMB Power Spectrum Explorer** [1]. This project was motivated by the need to bridge the gap between theoretical cosmology and hands-on parameter exploration—a skill essential for interpreting data from missions like Planck [2,3] and future experiments such as CMB-S4.

## Technical Implementation and Capabilities

The CMB Explorer is a production-ready Streamlit application that integrates CAMB (Code for Anisotropies in the Microwave Background) [4,5] to compute CMB power spectra with 21 adjustable cosmological parameters. These include the six primary ΛCDM parameters (H₀, Ωᵦh², Ωᶜh², τ, nₛ, Aₛ), along with advanced parameters for testing extended cosmologies: neutrino masses, spatial curvature, dark energy equation of state, tensor-to-scalar ratio, running of the spectral index, pivot scales, CMB temperature, helium fraction, and accuracy controls [1]. The application provides real-time visualization of temperature (TT), E-mode (EE), B-mode (BB), and cross-correlation (TE) power spectra, enabling systematic investigation of how each parameter affects the acoustic peak structure, damping tail, and polarization signatures. By implementing computational optimizations including caching and manual computation triggers, the tool achieves research-grade stability suitable for extensive parameter space exploration [1].

## Research Benefits and Applications

This tool has significantly enhanced my practical understanding of CMB physics by allowing me to reproduce key results from the Planck collaboration [2,3] and test theoretical predictions from inflationary models. Through systematic parameter variation, I have gained intuitive insights into degeneracies between cosmological parameters—for example, how changes in Ωᵦh² affect the relative heights of acoustic peaks, or how the optical depth τ suppresses small-scale power. The application's export functionality enables me to generate publication-quality data for further statistical analysis, while its educational features with comprehensive tooltips and physical interpretations have deepened my theoretical foundation. Moreover, the integration of matter power spectrum computation and gravitational lensing effects [6] provides a comprehensive framework for understanding structure formation alongside CMB observations.

## Future Research Directions

Building upon this foundation, my future research will focus on parameter estimation using Bayesian inference methods [7,8] to constrain cosmological models with observational data. I plan to extend the CMB Explorer to include Fisher matrix analysis for forecast experiments, comparison modes to test ΛCDM against alternative models (such as dynamical dark energy or modified gravity), and integration with large-scale structure data to break parameter degeneracies. Specifically, I am interested in investigating the implications of CMB lensing for neutrino mass constraints [3,9] and exploring how improved measurements of the primordial power spectrum running [10] can discriminate between inflationary scenarios. This project has equipped me with the computational skills, physical intuition, and research methodology necessary to contribute to next-generation CMB analysis pipelines and ultimately to our understanding of fundamental cosmology.

## Concluding Perspective

The development of the CMB Power Spectrum Explorer represents more than a technical achievement—it embodies my commitment to active, hands-on engagement with cosmological research and my belief that deep understanding comes from building tools that connect theory to observation. This project has taught me that modern cosmology demands not only theoretical knowledge but also computational proficiency, the ability to validate results against observational benchmarks, and the creativity to develop new approaches for extracting physical insights from complex datasets. As I pursue graduate studies, I bring this experience of independently developing research-grade software, the discipline to ensure reproducibility and accuracy in computational work, and the intellectual curiosity that drove me to explore every parameter's effect on the CMB power spectrum rather than simply accepting textbook results. My goal is to contribute to the field's ongoing efforts to constrain fundamental physics through precision cosmology, whether by improving parameter estimation techniques, developing novel statistical methods for model comparison, or helping to interpret data from next-generation experiments like Simons Observatory and CMB-S4. The skills and insights gained from this project have prepared me to tackle the challenges of PhD research, where theoretical understanding, computational expertise, and observational data must be seamlessly integrated to push the boundaries of our cosmological knowledge.

---

## References

[1] Chaulagain, S. (2025). *CMB Power Spectrum Explorer v2.0: An Interactive Tool for Cosmological Parameter Analysis*. GitHub Repository. https://github.com/[username]/CMB_Explorer_App

[2] Planck Collaboration, Aghanim, N., et al. (2020). "Planck 2018 results. VI. Cosmological parameters." *Astronomy & Astrophysics*, 641, A6. arXiv:1807.06209

[3] Planck Collaboration, Aghanim, N., et al. (2020). "Planck 2018 results. VIII. Gravitational lensing." *Astronomy & Astrophysics*, 641, A8. arXiv:1807.06210

[4] Lewis, A., Challinor, A., & Lasenby, A. (2000). "Efficient computation of CMB anisotropies in closed FRW models." *The Astrophysical Journal*, 538(2), 473-476. arXiv:astro-ph/9911177

[5] Howlett, C., Lewis, A., Hall, A., & Challinor, A. (2012). "CMB power spectrum parameter degeneracies in the era of precision cosmology." *Journal of Cosmology and Astroparticle Physics*, 2012(04), 027. arXiv:1201.3654

[6] Challinor, A., & Lewis, A. (2005). "Lensed CMB power spectra from all-sky correlation functions." *Physical Review D*, 71(10), 103010. arXiv:astro-ph/0502425

[7] Lewis, A., & Bridle, S. (2002). "Cosmological parameters from CMB and other data: A Monte Carlo approach." *Physical Review D*, 66(10), 103511. arXiv:astro-ph/0205436

[8] Lewis, A. (2013). "Efficient sampling of fast and slow cosmological parameters." *Physical Review D*, 87(10), 103529. arXiv:1304.4473

[9] Ade, P. A. R., et al. (BICEP2/Keck Array and Planck Collaborations). (2015). "Joint Analysis of BICEP2/Keck Array and Planck Data." *Physical Review Letters*, 114(10), 101301. arXiv:1502.00612

[10] Planck Collaboration, Aghanim, N., et al. (2020). "Planck 2018 results. V. CMB power spectra and likelihoods." *Astronomy & Astrophysics*, 641, A5. arXiv:1907.12875

---

## Single Paragraph Version (For Direct Copy-Paste into SOP)

My fascination with the Cosmic Microwave Background (CMB) stems from its unique position as a direct observational window into the early universe, merely 380,000 years after the Big Bang, where the CMB power spectrum encodes fundamental information about cosmological parameters that govern our universe's evolution, structure formation, and ultimate fate. To deepen my understanding of how these parameters shape observable CMB features, I developed an interactive web-based application called the CMB Power Spectrum Explorer [1], a production-ready Streamlit application that integrates CAMB (Code for Anisotropies in the Microwave Background) [4,5] to compute CMB power spectra with 21 adjustable cosmological parameters including the six primary ΛCDM parameters (H₀, Ωᵦh², Ωᶜh², τ, nₛ, Aₛ) along with advanced parameters for testing extended cosmologies such as neutrino masses, spatial curvature, dark energy equation of state, tensor-to-scalar ratio, running of the spectral index, pivot scales, CMB temperature, helium fraction, and accuracy controls. This tool has significantly enhanced my practical understanding of CMB physics by allowing me to reproduce key results from the Planck collaboration [2,3] and test theoretical predictions from inflationary models, gaining intuitive insights into degeneracies between cosmological parameters—for example, how changes in Ωᵦh² affect the relative heights of acoustic peaks, or how the optical depth τ suppresses small-scale power—while the application's export functionality enables me to generate publication-quality data for further statistical analysis. Building upon this foundation, my future research will focus on parameter estimation using Bayesian inference methods [7,8] to constrain cosmological models with observational data, extending the CMB Explorer to include Fisher matrix analysis for forecast experiments, comparison modes to test ΛCDM against alternative models (such as dynamical dark energy or modified gravity), and integration with large-scale structure data to break parameter degeneracies, with specific interest in investigating the implications of CMB lensing for neutrino mass constraints [3,9] and exploring how improved measurements of the primordial power spectrum running [10] can discriminate between inflationary scenarios. The development of this project represents more than a technical achievement—it embodies my commitment to active, hands-on engagement with cosmological research and my belief that deep understanding comes from building tools that connect theory to observation, teaching me that modern cosmology demands not only theoretical knowledge but also computational proficiency, the ability to validate results against observational benchmarks, and the creativity to develop new approaches for extracting physical insights from complex datasets. This project has equipped me with the computational skills, physical intuition, and research methodology necessary to contribute to next-generation CMB analysis pipelines and ultimately to our understanding of fundamental cosmology, preparing me to tackle the challenges of PhD research where theoretical understanding, computational expertise, and observational data must be seamlessly integrated to push the boundaries of our cosmological knowledge.

---

**Word Count:** ~650 words (4-paragraph version) | ~450 words (single paragraph version)  
**Format:** Ready for Statement of Purpose / Research Statement  
**Citation Style:** Standard astrophysics (author-year with arXiv)
