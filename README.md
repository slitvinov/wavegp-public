# wavegp-public

Public reproduction notebook for the **double shear layer** test of

> D. L. Brown & M. L. Minion, *Performance of Under-resolved Two-Dimensional
> Incompressible Flow Simulations*, J. Comput. Phys. **122** (1995) 165–183.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/slitvinov/wavegp-public/blob/main/brown.ipynb)

`brown.ipynb` builds the AMR incompressible projection solver, runs the
double shear layer at 64² and 128² (256² / 512² optional), and lays the
computed vorticity contours side-by-side with the paper's Fig. 2 (t=0.8) and
Fig. 3 (t=1.2).

The notebook is **self-contained**: the solver `main.c` and the table
generator `gen_table.py` are embedded inside it (written out with
`%%writefile`).  Only the paper reference panels are downloaded.

## Files

- `brown.ipynb` / `brown.py` — the notebook (jupytext-paired); `main.c` and
  `gen_table.py` are embedded as `%%writefile` cells
- `ref/` — reference panels cropped from Brown & Minion (1995), Figs. 2–3,
  reproduced here for direct visual comparison

Click the badge above to run it in Google Colab.
