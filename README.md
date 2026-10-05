<h1 align="center">FrontVeg V2</h1>

<p align="center">
  <b>Foreground-aware zero-shot plant trait segmentation in trellised crops<br>from a single side-view RGB image</b>
</p>

<p align="center">
  <a href="https://napari-hub.org/plugins/frontvegv2.html"><img src="https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/frontvegv2" alt="napari hub"></a>
  <a href="https://pypi.org/project/frontvegV2/"><img src="https://img.shields.io/pypi/v/frontvegV2.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/djaliloh/FrontVegV2/actions/workflows/ci.yml"><img src="https://github.com/djaliloh/FrontVegV2/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/djaliloh/FrontVegV2/main/docs/assets/images/examples_grid.jpg" alt="FrontVeg V2 segmentation examples on vine, apricot and apple trees" width="100%">
</p>

FrontVeg V2 is a [napari](https://napari.org) plugin and a Python package that segments plant organs (leaves, bunches, fruits) of the **front row** of trellised crops, without any training. It combines:

- **Depth-Anything-V2** monocular depth estimation, to separate the front row from the rows behind it;
- **SAM3** text-prompted segmentation, to detect every organ matching a word such as `leaf` or `grape`.

## Highlights

- **Zero-shot**: no annotation or training, change the organ with a text prompt.
- **Front row only**: automatic depth thresholding removes background rows and landscape.
- **High-resolution images**: tiled SAM3 inference with graph-based merging of objects cut by tile borders.
- **Two interfaces**: an interactive napari panel and a batch command line, sharing the same pipeline.
- **Measurements**: per-object areas exported to CSV, binary and instance masks exported to PNG.

## How it works

<p align="center">
  <img src="https://raw.githubusercontent.com/djaliloh/FrontVegV2/main/docs/assets/images/pipeline.svg" alt="FrontVeg V2 pipeline" width="100%">
</p>

Details in [How it works](docs/how-it-works.md).

## Quick start

**1. Install** (full guide: [docs/installation.md](docs/installation.md))

```bash
conda create -n frontveg python=3.10 -y && conda activate frontveg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

git clone --recursive https://github.com/djaliloh/FrontVegV2.git
cd FrontVegV2
pip install -e external/sam3
pip install -e .
```

**2. Download the checkpoints** into `checkpoints/`: [Depth-Anything-V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) (`depth_anything_v2_vitl.pth`) and [SAM3](https://huggingface.co/facebook/sam3) (`sam3.pt`, access on request). See [Step 4](docs/installation.md#step-4-download-the-checkpoints).

**3. Segment interactively**: run `napari`, open an image, then **Plugins > FrontVeg V2 Studio** and **Run Complete Pipeline**.

<p align="center">
  <img src="https://raw.githubusercontent.com/djaliloh/FrontVegV2/main/docs/assets/screenshots/napari-04-run.jpg" alt="FrontVeg V2 Studio in napari" width="85%">
</p>

**4. Or process folders**:

```bash
frontveg --input data/ --output results/ --prompt leaf
```

> **Video tutorial:** https://youtu.be/0sHq0k9mMcc

## Documentation

| Guide | Content |
|---|---|
| [Installation](docs/installation.md) | Environment, models, checkpoints, PyPI install |
| [napari plugin](docs/napari-guide.md) | Step-by-step guide with screenshots |
| [Batch processing](docs/batch-processing.md) | Command-line tools, outputs, Python API |
| [How it works](docs/how-it-works.md) | Depth thresholding, tiled SAM3, fusion |
| [Parameters](docs/parameters.md) | Defaults and tuning guide |
| [Troubleshooting](docs/troubleshooting.md) | Common errors and fixes |
| [Contributing](CONTRIBUTING.md) | Development setup, tests, conventions |
| [Changelog](CHANGELOG.md) | Release notes |

## Authors

**Abdoul Djalil Ousseini Hamza** ([ORCID](https://orcid.org/0009-0008-6172-4439)), **Herearii Metuarea** ([ORCID](https://orcid.org/0009-0008-2716-0617)), **Corentin Lothodé** ([ORCID](https://orcid.org/0000-0002-8209-317X)), **Morgane Roth** ([ORCID](https://orcid.org/0000-0002-5244-4215)), **Eric Duchêne** ([ORCID](https://orcid.org/0000-0003-2712-1892)), **Lionel Ley**, **David Alletru** ([ORCID](https://orcid.org/0009-0006-6238-6123)) and **David Rousseau**<sup>*</sup> ([ORCID](https://orcid.org/0000-0002-7935-1609))

<sup>*</sup> Project supervisor

## Citation

If you use FrontVeg V2 in your research, please cite:

```bibtex
@inproceedings{hamza2026foregroundaware,
  title     = {Foreground-Aware Zero-Shot Plant Trait Segmentation in Trellised Crops Using Side View Monocular {RGB} Images},
  author    = {Abdoul Djalil Ousseini Hamza and Herearii Metuarea and Corentin Lothodé and Morgane Roth and Eric Duchêne and Lionel LEY and David Allétru and David Marc ROUSSEAU},
  booktitle = {11th Workshop on Computer Vision in Plant Phenotyping and Agriculture},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Z24CpPVqcU}
}
```

The SoftwareX paper citation will be added on publication.

## Acknowledgements

<p align="center">
  <img src="https://raw.githubusercontent.com/djaliloh/FrontVegV2/main/docs/assets/images/partners.png" alt="Partners and funders" width="100%">
</p>

This work was supported by the French **Programme et Équipements Prioritaires de Recherche (PEPR) AgroEcoNum** ([pepr-agroeconum.fr](https://www.pepr-agroeconum.fr/)), the **France 2030** program through the Agence Nationale de la Recherche (ANR), and the European Union's **Horizon Europe** research and innovation programme (Grant No. 101094587, **PHENET**).

This research used computing resources from the **GLiCID Computing Facility** (Ligerien Group for Intensive Distributed Computing, [doi:10.60487/glicid](https://doi.org/10.60487/glicid), Pays de la Loire, France), as well as HPC and storage resources provided by **GENCI at IDRIS** on the Jean Zay supercomputer's H100 partition (Grant 2025-AD010115553R1).

The authors thank **Sirine Gharbi**, **Oumaima Karia**, **Justin Langlois**, **Paul Persello** and **Thomas Lebouc** for their help in annotating the image dataset used in this study.

## Contact

Imhorphen team, bioimaging research group, IRHS, UMR 1345, INRAE, 42 Rue Georges Morel, 49070 Beaucouzé, France

- David Rousseau, Professor: david.rousseau@univ-angers.fr
- Corentin Lothodé, Research Engineer: corentin.lothode@inrae.fr
- Herearii Metuarea, PhD student: herearii.metuarea@univ-angers.fr
- Abdoul Djalil Ousseini Hamza, Engineer: abdoul-djalil.ousseini-hamza@inrae.fr

Bugs and feature requests: [GitHub issues](https://github.com/djaliloh/FrontVegV2/issues).

## License

FrontVeg V2 is released under the [MIT License](LICENSE.txt). SAM3 and Depth-Anything-V2, used by FrontVeg V2, are subject to their own licenses and terms.
