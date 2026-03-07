# Radial-Cyclic Field Model (RCFM)

Author: Jeremy Erich Resch

This repository contains the full scientific production pipeline for the
Radial-Cyclic Field Model (RCFM). It includes the LaTeX manuscript source,
simulation code used to generate plots, bibliography, figure assets, and
versioned snapshots of the manuscript.

The repository is designed to make the scientific work reproducible and
transparent.

---

# Repository Purpose

The repository serves several roles:

• storage of the manuscript source  
• generation of figures used in the paper  
• reproducibility of computational results  
• archival of published versions  

Each published version of the paper is preserved together with the
exact figures and sources used to generate it.

---

# Repository Structure

```
.
├── document.tex
├── rcfm.bib
├── rcfm_simulation.py
│
├── figures (*.png)
│
├── pdf/
│   └── v/
│       ├── 1.1/
│       └── 1.2/
│
└── v/
    ├── 1.1/
    └── 1.2/
```

### Key Files

document.tex  
Main LaTeX manuscript.

rcfm.bib  
Bibliography database used by the manuscript.

rcfm_simulation.py  
Python script used to generate figures.

*.png  
Figures included in the manuscript.

*_corrected.png  
Post-processed figures used in the final paper.

---

# Building the Paper

Requirements

• LaTeX distribution (TeX Live recommended)  
• BibTeX or Biber  

Build using:

```
latexmk -pdf document.tex
```

Manual build:

```
pdflatex document.tex
bibtex document
pdflatex document.tex
pdflatex document.tex
```

The resulting PDF is placed in the working directory.

---

# Generating Figures

Figures are produced by the simulation script:

```
python3 rcfm_simulation.py
```

The script generates the plots used in the manuscript.

These include:

H(z) comparison  
background evolution  
CMB TT spectrum  
CMB EE spectrum  
growth function  
matter power spectrum  
parameter constraints  

---

# Versioning

The repository contains reproducible snapshots of each version
of the manuscript.

```
v/<version>/
```

Each version directory contains the exact files used for that release.

Compiled PDFs are stored in:

```
pdf/v/<version>/
```

Example:

```
pdf/v/1.2/rcfm.pdf
```

---

# Reproducibility

The repository includes all components needed to reproduce
the manuscript.

This includes:

• simulation code  
• generated figures  
• LaTeX source  
• bibliography  

The goal is that anyone can reproduce the paper by cloning
the repository and rebuilding the document.

---

# Citation

If you use, reference, or derive work from the Radial-Cyclic Field Model,
please cite the original work.

A citation metadata file is provided in `CITATION.cff`.

Example citation:

Resch, Jeremy Erich.  
Radial-Cyclic Field Model (RCFM).  
Version 1.2, 2026.

---

# Licensing

This repository uses two licenses.

Code license  
MIT License (see LICENSE_CODE)

Manuscript, model description, figures and documentation  
Creative Commons Attribution 4.0 (see LICENSE_PAPER)

Both licenses require attribution to the original author.

---

# Author

Jeremy Erich Resch

born in  Vienna, Austria 
Creator of the Radial-Cyclic Field Model (RCFM).
