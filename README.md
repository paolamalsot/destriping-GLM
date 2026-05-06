# destriping-GLM

Code and experiments for a GLM-based approach to reduce striping artifacts in 10x Genomics VisiumHD total-count images using nuclei segmentation. 

The repository includes reproducible pipelines for real-data benchmarks and synthetic simulations, plus analysis notebooks to generate figures and summaries.


---

## Motivation

10x Genomics VisiumHD enables spatial transcriptomics at ~2 μm² resolution but often exhibits **slide-specific, non-periodic striping artifacts** due to lane-width variability. These multiplicative row/column effects distort bin total counts and can bias downstream analyses.

---

## Method

![Method overview](assets/method_explanation.png)

We assume that the *true* size of the bin in row $i$ and column $j$ equals $h_i w_j$, where $h_i$ and $w_j$ are the corresponding horizontal and vertical lane widths, respectively. We further assume the total transcript concentration is homogeneous within nuclei, such that the expected value of total counts in bin $[i,j]$ is:

$$\mu_{ij} = c_{p(i,j)} \, h_i \, w_j$$


where $c_p$ is the total transcript concentration in nucleus $p$ and $p(i,j)$ denotes the nucleus to which bin $[i,j]$ belongs.

Finally, we model observed total counts $y_{ij}$ with a Negative Binomial distribution:

$$y_{ij} \sim \mathrm{NB}(\mu_{ij}, \phi) \,,$$

with mean $\mu_{ij}$ and dispersion parameter $\phi$.

### GLM parameterization

We fit nucleus concentrations and stripe factors jointly in a generalized linear modeling framework, with:

- **cross-validated regularization** on stripe factors $(\log(h_i), \log(w_j))$,
- **iterative dispersion estimation** for $\phi$,
- a correction step that converts the fitted parameters into a **destriped counts image**.

Implementation note: GLM fitting and regularized optimization are performed using **glum** (QuantCo):  
https://github.com/Quantco/glum

---

## Results (high level)

### Synthetic data (known ground truth)
- Improved stripe-factor estimation accuracy.
- Lower error in corrected counts compared to `bin2cell` and `bin2cell`-derived baselines.
- More accurate cell-typing.

### Public VisiumHD slides

On 4 datasets (mouse brain, mouse embryo, zebrafish head and human lymph node):
- Consistently lowers striping intensity.
- Better preserves biological structure present in global/large-scale count patterns.
- Avoids artifacts (e.g., macro-stripes/edge effects/reversed DGE effects) observed with sequential quantile normalization.

---

## Installation

This repo uses `pixi`.

```bash
pixi install
```
