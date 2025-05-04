# LUCID: Lightweight Unified Classifier for Intrusion Detection (Optimized)

LUCID is an optimized deep learning framework designed for real-time detection of DDoS attacks in network traffic. Originally proposed by Doriguzzi-Corin et al. (2020), this fork builds on their work and incorporates a set of targeted enhancements developed. These optimizations aim to reduce latency, overhead, and maintain high detection accuracy.

**Disclaimer**: This repository is a derivative of the original LUCID system. Proper credit is given below. The enhancements are part of an independent academic research project.

---

## 📄 Original Citation

Please cite the original paper if you use LUCID or this optimized version:

**R. Doriguzzi-Corin, S. Millar, S. Scott-Hayward, J. Martínez-del-Rincón and D. Siracusa**,  
*"Lucid: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection,"*  
*IEEE Transactions on Network and Service Management, vol. 17, no. 2, pp. 876–889, June 2020.*  
DOI: [10.1109/TNSM.2020.2971776](https://doi.org/10.1109/TNSM.2020.2971776)  

Funded by Horizon 2020 under grants no. 815141 (DECENTER), 830929 (CyberSec4Europe), and 833685 (SPIDER).

---

## Features & Optimizations

| # | Optimization              | Description                                                                 |
|---|---------------------------|-----------------------------------------------------------------------------|
| 1 | Optimized CNN Architecture | Uses `SeparableConv2D`, `BatchNormalization`, and `GlobalAveragePooling2D`. |
| 2 | Efficient Data Preprocessing | Speeds up PCAP parsing using `multiprocessing.Pool` and `joblib` caching.  |
| 3 | Hyperparameter Tuning     | Applies `RandomizedSearchCV` for optimizing LR, batch size, dropout, etc.  |
| 4 | Model Quantization        | Converts `.h5` models to `.tflite` for faster inference and smaller size.  |
| 5 | Threaded Live Prediction  | Captures and classifies packets in real-time using parallel threads.       |
| 6 | Colorized Output + Logging | Uses ANSI colors for terminal output and logs all predictions to file.     |

---



## Project Structure

lucid/
├── data/                |  # Datasets and parsed HDF5 files
├── models/              |  # Saved Keras and TFLite models
├── output/              | # Best model checkpoints and logs
├── sample-dataset/      | # Sample PCAPs for live prediction
├── lucid_cnn.py         | # Main training + evaluation script
├── live_predict.py      | # Threaded packet capture and prediction
├── utils/
│   ├── data_preprocessing.py  | # PCAP parsing, caching, and cleaning
│   ├── model_architectures.py | # CNN architectures (original and optimized)
│   ├── training.py            | # Training, tuning, early stopping
│   ├── inference.py           | # Inference + quantization
│   └── logger.py              | # Colored output + logging
├── requirements.txt
└── README.md

---



## Setup Instructions


This will:

Train the CNN using the optimized architecture

Tune hyperparameters using RandomizedSearchCV

Apply early stopping and save the best model

Quantize the best model to .tflite

Saved models are located in ./output/.

Predict from Live Traffic
```
python --predict_live --model ./output/best_model.tflite
```



## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/lucid.git
cd lucid
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
python lucid_cnn.py --train --dataset ./data/train.hdf5
```

This command will:

* Train the CNN using the optimized architecture defined in `utils/model_architectures.py`.
* Tune hyperparameters using `RandomizedSearchCV` for optimal performance.
* Apply early stopping to prevent overfitting and save the best model.
* Quantize the best trained model to a `.tflite` format for efficient inference.

Saved Keras models and their quantized `.tflite` versions, along with training logs, will be located in the `./output/` directory.

## 🧪 Predict from Live Traffic
bash python live_predict.py --model ./output/best_model.tflite
This script performs:

* Real-time packet sniffing from your network interface.
* Threaded prediction of potential DDoS attacks using the provided quantized `.tflite` model.
* Clear, colored console output indicating the classification of each processed packet.
* Detailed logging of predictions to `output/prediction_log.txt`.

## Example Output
Example Output

[DDoS Alert] DDoS Rate:__
[Normal] DDoS Rate:__
Log file: output/prediction_log.txt

## Model Performance (Validation)

| Metric      | Value   |
| ----------- | ------- |
| Accuracy    | 99.02%  |
| F1-Score    | 0.99    |
| Inference   | ~2 ms   |
| Model Size  | ~200 KB |
| Evaluated on | CIC-DDoS2019 using 80/20 split. |

## Tips & Troubleshooting

* **Dataset Class Balance:** Ensure your training dataset has a balanced representation of normal and attack traffic. Consider techniques like:
    * Undersampling the majority class.
    * Applying SMOTE (Synthetic Minority Over-sampling Technique) for minority oversampling.
    * Using class weights during training to account for imbalances.
* **Hyperparameter Tuning:** If you adapt Lucid to a new dataset or deployment environment, you might need to re-tune the hyperparameters in `lucid_cnn.py` for optimal performance.
* **Model Selection:** Experiment with different CNN architectures in `utils/model_architectures.py` to find the best fit for your specific needs and resource constraints.

## License and Acknowledgement
# License

This project is licensed under the MIT License. See LICENSE for full details.
This work incorporates and extends the official LUCID project (Apache License 2.0) and is intended for academic and non-commercial research use.

# Acknowledgements
Thanks to the original authors of LUCID for their foundational work in lightweight, deep learning-based DDoS detection.
This optimized version was developed as part of a university thesis focused on low-latency packet classification.
