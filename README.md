# Multimodal-Isolated-Italian-Sign-Language-Recognition

This repository is the official implementation of the following paper.

Paper Title: **[FusionEnsemble-Net: An Attention-Based Ensemble of Spatiotemporal Networks for Multimodal Sign Language Recognition](https://arxiv.org/abs/2508.09362)**
<br/>
[Md. Milon Islam](https://scholar.google.com/citations?user=ABC_LOAAAAAJ&hl=en), [Md Rezwanul Haque](https://rezwanh001.github.io/),  [S M Taslim Uddin Raju](https://scholar.google.com/citations?user=ToadRS8AAAAJ&hl=en), [Hamdi Altaheri](https://scholar.google.com/citations?user=UkfsK6EAAAAJ&hl=en), [Lobna Nassar](https://scholar.google.com/citations?user=3vcIuocAAAAJ&hl=en), [Fakhri Karray](https://uwaterloo.ca/scholar/karray)
<br/>

#### ***Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Honolulu, Hawaii, USA. 1st MSLR Workshop 2025. Copyright 2025 by the author(s).***

[![arXiv](https://img.shields.io/badge/arXiv-2508.08093-b31b1b)](https://arxiv.org/abs/2508.09362)


![](assets/FusionEnsemble-Net.png)


## Abstract
> Accurate recognition of sign language in healthcare communication poses a significant challenge, requiring frameworks that can accurately interpret complex multimodal gestures. To deal with this, we propose FusionEnsemble-Net, a novel attention-based ensemble of spatiotemporal networks that dynamically fuses visual and motion data to enhance recognition accuracy. The proposed approach processes RGB video and range Doppler map radar modalities synchronously through four different spatiotemporal networks. For each network, features from both modalities are continuously fused using an attention-based fusion module before being fed into an ensemble of classifiers. Finally, the outputs of these four different fused channels are combined in an ensemble classification head, thereby enhancing the model’s robustness. Experiments demonstrate that FusionEnsemble-Net outperforms state-of-the-art approaches with a test accuracy of 99.44% on the large-scale MultiMeDaLIS dataset for Italian Sign Language. Our findings indicate that an ensemble of diverse spatiotemporal networks, unified by attention-based fusion, yields a robust and accurate framework for complex, multimodal isolated gesture recognition tasks. The source code is available at: https://github.com/rezwanh001/Multimodal-Isolated-Italian-Sign-Language-Recognition.

---
- ***Team Name:*** **CPAMI (UW)**
- **📊 For reference, the best accuracy of our method was `99.365%` on the validation set and `99.444%` on the test set.**
---

### 📖 Citation:
- If you find this project useful for your research, please cite [this paper](https://arxiv.org/abs/2508.09362)

```bibtex
@article{islam2025fusionensemble,
  title={FusionEnsemble-Net: An Attention-Based Ensemble of Spatiotemporal Networks for Multimodal Sign Language Recognition},
  author={Islam, Md Milon and Haque, Md Rezwanul and Raju, SM and Karray, Fakhri},
  journal={arXiv preprint arXiv:2508.09362},
  year={2025}
}
```
---

1st Multimodal Isolated Italian Sign Language Recognition C. using RGB and Radar-RDM Data from the [MultiMeDaLIS Dataset](https://www.kaggle.com/competitions/iccv-mslr-2025-track-2/data) (Mineo et al., 2024). This track presents a sign language recognition task on our multimodal dataset, featuring RGB videos and 60 GHz radar range-Doppler maps, and including 126 Italian Sign Language gestures (100 medical terms + 26 letters) across 205 expert sessions.

---

## Running the Model

```
python train.py 
```

## Generating the submission file

```
python submission.py 
```

## Result Overview:
### Model Performance Results

Here are the performance metrics for various models, including individual architectures and an ensemble approach.

| Model                       | Validation Acc | Test Acc |
|-----------------------------|----------------|----------|
| TwoStreamCNNLSTM (3D ResNet)           | 0.96575        | 0.96575  |
| AdvancedTwoStreamModel (MC3)      |                |          |
| &nbsp;&nbsp;&nbsp;&nbsp;- Run 1 | 0.98594        | 0.98594  |
| &nbsp;&nbsp;&nbsp;&nbsp;- Run 2 | 0.98752        | 0.99126  |
| &nbsp;&nbsp;&nbsp;&nbsp;- Run 3 | 0.98662        | 0.98994  |
| &nbsp;&nbsp;&nbsp;&nbsp;- Run 4 | 0.98956        | 0.99060  |
| UltraAdvancedTwoStreamModel (R(2+1)D) | 0.96938        | 0.97341  |
| SwinTwoStreamModel (Swin-B)          | 0.94240        | 0.94417  |
| **Ensemble All Model (FusionEnsemble-Net)** | **0.99365** | **0.99444** |

**Note:** The ensemble model combines TwoStreamCNNLSTM, AdvancedTwoStreamModel, UltraAdvancedTwoStreamModel, and SwinTwoStreamModel.

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3) - see the [LICENSE](LICENSE) file for details.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

© 2025 Md Rezwanul Haque and Contributors.

