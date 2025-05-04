# Optimized_LUCID

LUCID: Lightweight Unified Classifier for Intrusion Detection (Optimized)
LUCID is an optimized deep learning framework designed for real-time detection of DDoS attacks in network traffic. Originally proposed by Doriguzzi-Corin et al. (2020), this fork builds on their work and incorporates a set of targeted enhancements developed as part of a university thesis. These optimizations aim to reduce latency, improve throughput, and maintain high detection accuracy.

⚠️ Disclaimer: This repository is a derivative of the original LUCID system. Proper credit is given below. The enhancements are part of an independent academic research project.

📄 Original Citation
Please cite the original paper if you use LUCID or this optimized version:

R. Doriguzzi-Corin, S. Millar, S. Scott-Hayward, J. Martínez-del-Rincón and D. Siracusa,
"Lucid: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection,"
in IEEE Transactions on Network and Service Management, vol. 17, no. 2, pp. 876–889, June 2020.
DOI: 10.1109/TNSM.2020.2971776

Funded by Horizon 2020 under grants no. 815141 (DECENTER), 830929 (CyberSec4Europe), and 833685 (SPIDER).

🚀 Features & Optimizations
#	Optimization	Description
1	Optimized CNN Architecture	Uses SeparableConv2D, BatchNormalization, and GlobalAveragePooling2D for a lightweight yet accurate model.
2	Efficient Data Preprocessing	Speeds up PCAP parsing with multiprocessing.Pool and joblib caching.
3	Hyperparameter Tuning	Utilizes RandomizedSearchCV for learning rate, batch size, dropout, and kernel size optimization.
4	Model Quantization	Converts .h5 models to .tflite for faster and smaller inference-ready models.
5	Threaded Live Prediction	Performs real-time traffic capture and prediction in parallel threads.
6	Colorized Output + Logging	Displays colored labels (e.g., red for DDoS) and saves logs to file for traceability.

🧩 Project Structure
graphql
Copy code
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
⚙️ Setup Instructions
1. Clone the Repository
bash
Copy code
git clone https://github.com/your-username/lucid.git
cd lucid
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
3. Prepare Dataset
Place your .pcap or .hdf5 training files into the data/ directory. Use utils/data_preprocessing.py to parse raw .pcap files if needed.

🏋️‍♂️ Train the Model
bash
Copy code
python lucid_cnn.py --train --dataset ./data/train.hdf5
This will:

Train the CNN using optimized architecture

Tune hyperparameters with RandomizedSearchCV

Apply early stopping and save the best model

Quantize the model to .tflite

Saved models are located in ./output/.

🧪 Predict from Live Traffic
bash
Copy code
python live_predict.py --model ./output/best_model.tflite
This performs:

Real-time packet sniffing

Threaded DDoS prediction with quantized model

Colored and logged output

📊 Example Output
plaintext
Copy code
[DDoS Alert] 192.168.1.2 → 10.0.0.5 | Packet classified as DDoS ⚠️
[Normal]     10.0.0.5 → 192.168.1.2 | Packet classified as Normal ✅
Log file: output/prediction_log.txt

🧠 Model Performance (Validation)
Metric	Value
Accuracy	99.02%
F1-Score	0.99
Inference	~2 ms
Model Size	~200 KB (.tflite)

Evaluated on CIC-DDoS2019 using 80/20 split.

📌 Tips & Troubleshooting
Ensure dataset class balance. If needed, try:

Undersampling the majority class

Using SMOTE for oversampling

Applying class weights during training

Adjust learning rate, dropout, and kernel size if porting to new datasets or hardware.

📜 License
This project is licensed under the MIT License.
See LICENSE for details.

This work incorporates and extends the official LUCID project (Apache License 2.0) and is intended for academic and non-commercial research use.

🙏 Acknowledgements
Thanks to the original authors of LUCID for their foundational work in lightweight, deep learning-based DDoS detection.
This optimized version was developed for a university thesis focused on low-latency packet classification.
