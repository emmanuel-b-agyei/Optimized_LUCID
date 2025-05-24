# LUCID: Lightweight Unified Classifier for Intrusion Detection (Optimized)

LUCID is an optimized deep learning framework designed for real-time detection of DDoS attacks in network traffic. Originally proposed by Doriguzzi-Corin et al. (2020), this fork builds on their work and incorporates a set of targeted enhancements developed. These optimizations aim to reduce latency, overhead, and maintain high detection accuracy.

**Disclaimer**: This repository is a derivative of the original LUCID system: DOI: [10.1109/TNSM.2020.2971776](https://doi.org/10.1109/TNSM.2020.2971776). Partially funded by the European Union’s Horizon 2020 Research and Innovation Programme under grant agreements no. 815141 (DECENTER project), 830929 (CyberSec4Europe project) and n. 833685 (SPIDER project).

The enhancements are part of an independent academic research project.  *

---

## Features & Optimizations

| # | Optimization              | Description                                                                 |
|---|---------------------------|-----------------------------------------------------------------------------|
| 1 | Optimized CNN Architecture | Uses `SeparableConv2D`, `BatchNormalization`, and `GlobalAveragePooling2D`. |
| 2 | Efficient Data Preprocessing | Speeds up PCAP parsing using `multiprocessing.Pool` and `joblib` caching.  |
| 3 | Hyperparameter Tuning     |  Auto-tune model parameter using Optuna hyperparameter optimization framework  |
| 4 | Model Quantization        | Converts `.h5` models to `.tflite` for faster inference and smaller size.  |
| 5 | Threaded Live Prediction  | Captures and classifies packets in real-time using parallel threads.       |
| 6 | Colorized Output + Logging | Uses ANSI colors for terminal output and logs all predictions to file.     |

---


## Project Structure
```graphql

lucid/
├── data/                   # Datasets and parsed HDF5 files
├── models/                 # Saved Keras and TFLite models
├── output/                 # Best model checkpoints and logs
├── sample-dataset/         # Sample PCAPs for live prediction
├── lucid_cnn.py            # Main training + evaluation + Threaded packet capture and prediction
├── data_dataset_parser.py  # PCAP parsing, caching, and cleaning
├── requirements.txt
└── README.md
```
---



## Setup Instructions

1. Clone the Repository

```bash
git clone https://github.com/emmanuel-b-agyei/Optimized_LUCID.git
```

2. Install Dependencies
```
pip install -r requirements.txt
```

4. Prepare Dataset
   
Place your .pcap or .hdf5 training files into the data/ directory.
Use utils/data_preprocessing.py to convert raw PCAPs if needed.

5. Train the Model
```
python lucid_cnn.py --train --dataset ./<dataset-name>/
```
This command will:

* Train the CNN using the optimized architecture.
* Tune hyperparameters using Optuna for optimal performance.
* Early stopping with Optuna to prevent overfitting and save the best model.
* Quantize the best-trained model to a `.tflite` format for efficient inference.

Saved Keras models and their quantized `.tflite` versions, along with training logs, will be located in the `./output/` directory.


6. Test the the Model
```
python lucid_cnn.py --predict ./sample-dataset/ --model ./output/<best_model_name>_quantized.tflite
```

7.  Predict from Live Traffic
Predict from Live Traffic
```
python lucid_cnn.py --predict_live <network_interface_name> --model ./output/<best_model_name>_quantized.tflite
```
NB: You have to give administrative privilege to capture live packets

This will perform:

* Real-time packet sniffing from your network interface.
* Threaded prediction of potential DDoS attacks using the provided quantized `.tflite` model.
* Clear, colored console output indicating the classification of each processed packet.
* Detailed logging of predictions to `output/prediction_log.txt`.


## Model Performance (Validation)

| Metric      | Value   |
| ----------- | ------- |
| Accuracy    | 99.34%  |
| F1-Score    | 99.35    |
| Trainging Time| 88.65 s |
| Inference   | ~ 2 s   |
| VM spec | Intel Xeon 16 cores, 32GB / 254GB storage|
| Evaluated on | CIC-DDoS2019 |

## Tips & Troubleshooting

* **Dataset Class Balance:** Ensure your training dataset has a balanced representation of normal and attack traffic. Consider techniques like:
    * Undersampling the majority class.
    * Applying SMOTE (Synthetic Minority Over-sampling Technique) for minority oversampling.
    * Using class weights during training to account for imbalances.

## Acknowledgements
Thanks to the original authors of LUCID for their foundational work in lightweight, deep learning-based DDoS detection.
This optimized version was developed as part of a university thesis focused on low-latency and overhead packet classification.
