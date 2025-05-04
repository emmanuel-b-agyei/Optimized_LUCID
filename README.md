# LUCID: Lightweight Unified Classifier for Intrusion Detection (Optimized)

LUCID is an optimized deep learning framework designed for real-time detection of DDoS attacks in network traffic. Originally proposed by Doriguzzi-Corin et al. (2020), this fork builds on their work and incorporates a set of targeted enhancements developed as part of a university thesis. These optimizations aim to reduce latency, improve throughput, and maintain high detection accuracy.

> ⚠️ **Disclaimer**: This repository is a derivative of the original LUCID system. Proper credit is given below. The enhancements are part of an independent academic research project.

---

## 📄 Original Citation

Please cite the original paper if you use LUCID or this optimized version:

**R. Doriguzzi-Corin, S. Millar, S. Scott-Hayward, J. Martínez-del-Rincón and D. Siracusa**,  
*"Lucid: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection,"*  
*IEEE Transactions on Network and Service Management, vol. 17, no. 2, pp. 876–889, June 2020.*  
DOI: [10.1109/TNSM.2020.2971776](https://doi.org/10.1109/TNSM.2020.2971776)  

Funded by Horizon 2020 under grants no. 815141 (DECENTER), 830929 (CyberSec4Europe), and 833685 (SPIDER).

---

## 🚀 Features & Optimizations

| # | Optimization              | Description                                                                 |
|---|---------------------------|-----------------------------------------------------------------------------|
| 1 | Optimized CNN Architecture | Uses `SeparableConv2D`, `BatchNormalization`, and `GlobalAveragePooling2D`. |
| 2 | Efficient Data Preprocessing | Speeds up PCAP parsing using `multiprocessing.Pool` and `joblib` caching.  |
| 3 | Hyperparameter Tuning     | Applies `RandomizedSearchCV` for optimizing LR, batch size, dropout, etc.  |
| 4 | Model Quantization        | Converts `.h5` models to `.tflite` for faster inference and smaller size.  |
| 5 | Threaded Live Prediction  | Captures and classifies packets in real-time using parallel threads.       |
| 6 | Colorized Output + Logging | Uses ANSI colors for terminal output and logs all predictions to file.     |

---

## 🧩 Project Structure

lucid/
├── data/                         # Datasets and parsed HDF5 files
├── models/                       # Saved Keras and TFLite models
├── output/                       # Best model checkpoints and logs
├── sample-dataset/              # Sample PCAPs for live prediction
├── lucid_cnn.py                 # Main training + evaluation script
├── live_predict.py              # Threaded packet capture and prediction
├── utils/
│   ├── data_preprocessing.py    # PCAP parsing, caching, and cleaning
│   ├── model_architectures.py   # CNN architectures (original and optimized)
│   ├── training.py              # Training, tuning, early stopping
│   ├── inference.py             # Inference + quantization
│   └── logger.py                # Colored output + logging
├── requirements.txt
└── README.md
