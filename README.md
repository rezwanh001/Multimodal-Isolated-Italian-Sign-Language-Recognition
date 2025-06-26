# Multimodal-Isolated-Italian-Sign-Language-Recognition
---
---
1st Multimodal Isolated Italian Sign Language Recognition C. using RGB and Radar-RDM Data from the [MultiMeDaLIS Dataset](https://www.kaggle.com/competitions/iccv-mslr-2025-track-2/data) (Mineo et al., 2024). This track presents a sign language recognition task on our multimodal dataset, featuring RGB videos and 60 GHz radar range-Doppler maps, and including 126 Italian Sign Language gestures (100 medical terms + 26 letters) across 205 expert sessions.

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
| TwoStreamCNNLSTM            | 0.96575        | 0.96575  |
| AdvancedTwoStreamModel      |                |          |
| &nbsp;&nbsp;&nbsp;&nbsp;- Run 1 | 0.98594        | 0.98594  |
| &nbsp;&nbsp;&nbsp;&nbsp;- Run 2 | 0.98752        | 0.99126  |
| &nbsp;&nbsp;&nbsp;&nbsp;- Run 3 | 0.98662        | 0.98994  |
| &nbsp;&nbsp;&nbsp;&nbsp;- Run 4 | 0.98956        | 0.99060  |
| UltraAdvancedTwoStreamModel | 0.96938        | 0.97341  |
| SwinTwoStreamModel          | 0.94240        | 0.94417  |
| **Ensemble All Model** | **0.99365** | **0.99444** |

**Note:** The ensemble model combines TwoStreamCNNLSTM, AdvancedTwoStreamModel, UltraAdvancedTwoStreamModel, and SwinTwoStreamModel.