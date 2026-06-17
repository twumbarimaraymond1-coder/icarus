# Literature survey — heat partition & stochastic prediction

Scope: the two modelling directions from the 2026-06-17 supervisor meeting —
(A) **heat flux partitioning** into physical mechanisms, and (B) **stochastic /
probabilistic** prediction with calibrated uncertainty. Plus a short note on
(C) **SPOD**, the frequency-resolved decomposition that feeds both.

This is a working survey to scope the summer work and seed the paper's related-
work section, not an exhaustive review. Verify and read the primary sources
before citing.

---

## A. Heat flux partitioning

### The canonical model (RPI / Kurul–Podowski)
Kurul & Podowski (1990, RPI) split the wall heat flux during subcooled boiling
into **three** components:

- **q_c — single-phase convection**: liquid convection over the wall area *not*
  influenced by bubbles.
- **q_e — evaporation**: heat carried off as latent heat forming vapour
  (microlayer + superheated-layer evaporation).
- **q_q — quenching / transient conduction**: cold liquid rewetting the wall
  during the *waiting time* after a bubble departs, re-forming the thermal
  boundary layer.

Total: `q_w = q_c + q_e + q_q`. Closure needs three empirical inputs:
**active nucleation site density**, **bubble departure diameter**, and
**departure frequency**.

**Known limitations** (directly relevant to why an ML/data approach is
interesting): bubbles are assumed to lift off without sliding (contradicts
observation); the partition relies on empirical correlations that don't
transfer well across fluids/pressures/surfaces; the area-of-influence factor
is uncertain.

### Modern extensions
- **Five-component partitioning models** (e.g. Int. J. Multiphase Flow, 2022)
  add sliding-bubble and convection-enhancement terms.
- **Mechanistic models from bubble dynamics** (Int. J. Heat Mass Transfer,
  2021) derive the partition from first-principles bubble growth rather than
  fitted correlations.
- Work on **NaCl solutions vs pure water** (IJHMT 2019) shows how strongly the
  partition depends on fluid properties — i.e. why a single correlation fails.

### IR-thermography-based partitioning (most relevant to icarus)
This is the cluster closest to your data — partition measured *directly* from
through-substrate IR, not assumed from correlations:

- **DEPIcT** (Detection of Phase by IR Thermography): measures wetted-area
  fraction from the IR emission contrast between wet and dry wall regions.
- **Synchronous high-speed visual + IR partitioning** (IJHMT 2024): a method to
  partition boiling mechanisms by combining through-substrate high-speed video
  with IR — arguably the current state of the art and a direct methodological
  comparator for your approach.
- **High-speed IR thermometry of pool boiling** (IJHMT 2021): extracts
  nucleation site density, wait/growth times, bubble footprint radius, then
  quantifies microlayer-evaporation vs rewetting vs convection contributions.
- Microlayer-evaporation studies (IJHMT 2024) show its contribution rises
  monotonically with heat flux via nucleation-site density and departure
  frequency — a physical prior your model could be checked against.

**Implication for icarus:** rather than predicting total `q` in one shot,
predict each physical component (or learn POD/SPOD modes that map onto them).
The IR-partition literature gives both (a) ground-truth methods to generate
per-mechanism targets, and (b) physical sanity checks (e.g. microlayer share ↑
with flux). The novelty hook: *data-driven partition prediction* vs the
correlation-closed RPI model.

---

## B. Stochastic / probabilistic prediction

Boiling is intrinsically stochastic (random nucleation-site activation), so a
point estimate of `q` is arguably the wrong object — a *distribution* is more
faithful, and uncertainty is essential for any CHF-margin / safety use.

### The active sub-field: ML + UQ for critical heat flux (CHF)
CHF prediction with uncertainty is the most developed probabilistic-boiling-ML
area (driven by nuclear safety), and is the best template to borrow from:

- **Bayesian neural networks for CHF** (Applied Thermal Engineering 2025): a
  unified BNN giving mean + predictive uncertainty.
- **Physics-based hybrid ML with UQ** (arXiv:2502.19357 / ATE 2025): wraps
  empirical correlations (Biasi, Bowring) with three UQ techniques — **deep
  ensembles, Bayesian NNs, deep Gaussian processes**. Finding: hybrid
  (physics + ML) beats pure ML and resists data scarcity. Best config (Biasi
  hybrid DNN ensemble) reached ~1.85 % mean abs. relative error with stable
  uncertainty.
- **Probabilistic framework for vertical tubes** (Nuclear Technology 2026):
  BNN + **NGBoost** (Natural Gradient Boosting) on ~24.6k points predicting
  mean and uncertainty.
- **UQ study of physics-informed ML for CHF** (Progress in Nuclear Energy
  2024): systematic comparison of UQ methods.

### Candidate techniques for icarus (in rough order of effort)
1. **Quantile regression / NGBoost** — cheapest; predict prediction intervals
   directly. Good first probabilistic baseline.
2. **Deep ensembles** — train N nets, use spread as epistemic uncertainty.
   Simple, strong, parallelises. The CHF literature's most robust performer.
3. **Mixture Density Networks (MDN)** — output a full distribution per pixel;
   natural for multi-modal boiling behaviour.
4. **Bayesian NNs / MC-dropout** — principled posterior, more finicky.
5. **Gaussian processes / deep GPs** — gold-standard calibration, but scaling
   to millions of pixel-timesteps needs sparse/variational approximations.

**Key evaluation point:** report **calibration** (e.g. reliability diagrams,
PICP/coverage), not just accuracy — the whole value of going stochastic is
*honest* uncertainty. Separate **aleatoric** (boiling randomness, irreducible)
from **epistemic** (model ignorance, reducible with data) uncertainty; the
former is the interesting physics, the latter tells you where more data helps.

**Product link:** the immersion-cooling CHF-warning thesis *requires* calibrated
uncertainty — "margin to CHF is X with confidence Y." This direction is the
technical foundation of that product, not just an academic nicety.

---

## C. SPOD (feeds A and B)

- **Towne, Schmidt & Colonius (JFM 2018)** — the canonical reference. SPOD
  modes are eigenvectors of the cross-spectral density at each frequency;
  each mode oscillates at a single frequency. Relationship to DMD and
  resolvent analysis is established there.
- Reference implementation: `SpectralPOD/spod_matlab` (GitHub) — useful to
  cross-check the icarus `SPOD` implementation against.
- **Gap / opportunity:** SPOD is mature in compressible flows, jets, and
  boundary layers, but **applications to boiling / two-phase flow are sparse**.
  Applying SPOD to flow-boiling IR data to isolate nucleation vs departure vs
  microlayer timescales appears to be largely open ground — a genuine novelty
  angle, and it dovetails with the heat-partition direction (frequency bands ≈
  physical mechanisms).

---

## How the three connect (the summer thesis)

> SPOD resolves the boiling field into frequency-coherent structures →
> those structures map onto physical heat-transfer mechanisms (heat partition)
> → each mechanism is predicted with calibrated uncertainty (stochastic model).

Each step also closes a specific reviewer gap in the current result:
SPOD/partition → interpretability & cross-fluid generalisation;
stochastic → honesty about boiling's inherent randomness + CHF-margin utility.

---

## Sources

- [Five-component wall boiling heat flux partitioning model (IJMF 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0301932222002671)
- [Mechanistic wall heat flux partitioning from bubble dynamics (IJHMT 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0017931021003987)
- [Wall heat flux partitioning, NaCl vs water (IJHMT 2019)](https://www.sciencedirect.com/science/article/abs/pii/S0017931019364440)
- [On the Modeling of Wall Heat Flux Partitioning in Subcooled Flow Boiling (ResearchGate)](https://www.researchgate.net/publication/306079545_On_the_Modeling_of_Wall_Heat_Flux_Partitioning_in_Subcooled_Flow_Boiling)
- [Direct experimental measurement of wall heat flux partitioning (IJHMT 2018)](https://www.sciencedirect.com/science/article/abs/pii/S0017931018306458)
- [Wetted area fraction by IR thermography (DEPIcT) (Nucl. Eng. Des. 2013)](https://www.sciencedirect.com/science/article/abs/pii/S0029549313003233)
- [IR thermographic investigation of nucleate pool boiling at high heat flux (2015)](https://www.sciencedirect.com/science/article/abs/pii/S0140700715003138)
- [Heat transfer mechanisms in pool boiling by high-speed IR thermometry (IJHMT 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0017931021001095)
- [Partitioning boiling mechanisms via synchronous visual + IR (IJHMT 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0017931024003478)
- [Microlayer evaporation during bubble growth (IJHMT 2024)](https://www.sciencedirect.com/science/article/pii/S0017931024006914)
- [Physics-Based Hybrid ML for CHF with UQ (arXiv:2502.19357)](https://arxiv.org/abs/2502.19357)
- [Bayesian NN for CHF prediction with UQ (Applied Thermal Engineering 2025)](https://www.sciencedirect.com/science/article/abs/pii/S1359431125020101)
- [Probabilistic ML for CHF in vertical tubes with UQ (Nuclear Technology 2026)](https://www.tandfonline.com/doi/full/10.1080/00295450.2026.2629142)
- [UQ study of physics-informed ML for CHF (Prog. Nucl. Energy 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0149197024000477)
- [Towne, Schmidt & Colonius, SPOD and its relationship to DMD and resolvent analysis (JFM 2018)](https://www.semanticscholar.org/paper/Spectral-proper-orthogonal-decomposition-and-its-to-Towne-Schmidt/f7437a94166a3a51264e5d1615b9b0785107b71f)
- [spod_matlab reference implementation (GitHub)](https://github.com/SpectralPOD/spod_matlab)
