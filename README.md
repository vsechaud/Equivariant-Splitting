## Equivariant Splitting: Self-supervised learning from incomplete data<br><sub>Official PyTorch implementation</sub>

<p align="center"><img src="assets/Inpainting_Measurement.png" alt="Inpainting Measurement" style="width: 24%;"><img src="assets/Inpainting_EQ_ES.png" alt="Inpainting EQ ES" style="width: 24%;"><img src="assets/MRIx8_IDFT.png" alt="MRI×8 IDFT" style="width: 24%;"><img src="assets/MRIx8_EQ_ES.png" alt="MRI×8 EQ ES" style="width: 24%;"></p>

<p align="center">ES reconstructions from incomplete measurements (Inpainting, MRI)</p>

**Equivariant Splitting: Self-supervised learning from incomplete data**<br>

Abstract: *Self-supervised learning for inverse problems allows to train a reconstruction network from noise and/or incomplete data alone. These methods have the potential of enabling learning-based solutions when obtaining ground-truth references for training is expensive or even impossible. In this paper, we propose a new self-supervised learning strategy devised for the challenging setting where measurements are observed via a single incomplete observation model. We introduce a new definition of equivariance in the context of reconstruction networks, and show that the combination of self-supervised splitting losses and equivariant reconstruction networks results in the same minimizer in expectation as the one of a supervised loss. Through a series of experiments on image inpainting, accelerated magnetic resonance imaging, and compressive sensing, we demonstrate that the proposed loss achieves state-of-the-art performance in settings with highly rank-deficient forward models.*

### Experiments

Equivariant splitting is evaluated on different imaging modalities: compressive sensing, image inpainting and accelerated MRI. The code and instructions to reproduce the results for compressive sensing and inpainting are available [here](cs_inp) and for MRI [here](mri).

### Acknowledgment

[![SSIBench](https://img.shields.io/badge/GitHub-SSIBench-blue.svg)](https://github.com/Andrewwango/ssibench)
[![DeepInverse](https://img.shields.io/github/stars/deepinv/deepinv?label=DeepInverse)](https://deepinv.github.io/deepinv)

This work makes use of the efficient forward operators and training losses in DeepInverse and the convenient MRI training and comparison features of SSIBench.
