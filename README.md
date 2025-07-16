# Multimodal-Isolated-Italian-Sign-Language-Recognition

Paper Title: **FusionEnsemble-Net: An Attention-Based Ensemble of Spatiotemporal Networks for Multimodal Sign Language Recognition**

#### ***Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), Honolulu, Hawaii, USA. 1st MSLR Workshop 2025. Copyright 2025 by the author(s).***

---
- ***Team Name:*** **CPAMI (UW)**
- **📊 For reference, the best accuracy of our method was `99.365%` on the validation set and `99.444%` on the test set.**
---

### 📖 Citation:
- If you find this project useful for your research, please cite [this paper](https://arxiv.org/abs/****.*****)

```bibtex
@inproceedings{haque2025signer,
    title={FusionEnsemble-Net: An Attention-Based Ensemble of Spatiotemporal Networks for Multimodal Sign Language Recognition},
    author={Islam, Md. Milon and Haque, Md Rezwanul and Raju, S M Taslim Uddin and Karray, Fakhri},
    journal = {arXiv preprint arXiv:****.*****},
    year = {2025}
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

This project is licensed under the [MIT License](LICENSE).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
