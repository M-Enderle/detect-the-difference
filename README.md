# 💊 Detect the Difference

<div align="center">
  
![Pill Detection Banner](https://user-images.githubusercontent.com/73176174/171226130-2dd6c7f1-c9fb-45b3-ac64-e9013a42d5fd.png)

**An advanced pill detection algorithm developed for the Rohde & Schwarz Engineering Competition 2022**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![YOLOv5](https://img.shields.io/badge/Model-YOLOv5-brightgreen?style=flat-square)](https://github.com/ultralytics/yolov5)
[![Made with](https://img.shields.io/badge/Made%20with-❤️-red?style=flat-square)](https://github.com/M-Enderle/detect-the-difference)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

## 🎯 Challenge Overview

This project was developed for the **Rohde & Schwarz Engineering Competition 2022**, where our team successfully reached the finals. The challenge required developing an algorithm that could:

- 🔍 Automatically detect pills and empty blisters in microwave images
- 🧪 Pass tests against unknown images
- 🚀 Demonstrate generalization capability through advanced tasks
- 🎥 Present our solution through a compelling video pitch

## 💡 Our Solution

We implemented a **YOLOv5-based approach** for this object detection challenge. Our solution:

- Identifies pills with high precision in various scenarios
- Works efficiently with minimal processing time
- Achieves exceptional accuracy on test datasets
- Generalizes well to unseen data

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| 🎯 Anomaly Detection Accuracy | **99.21%** |
| 📈 Average Sample Accuracy | **99.74%** |
| 🔍 Distance Precision | **96.38%** |
| ⚡ Prediction Time | **8.79s** |
| 📝 Code Quality (pylint) | **100.00%** |

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r code_submission/requirements.txt
```

### Running the Detection

```python
# Example usage
python code_submission/detect_single.py --image_path path/to/your/image.jpg
```

## 🏗️ Project Structure

- **`code_submission/`**: Contains the core detection algorithm and model
- **`ingestion_program/`**: Handles data preprocessing and model input
- **`scoring_program/`**: Evaluation metrics and performance assessment

## 👨‍💻 Team Members

<div align="center">
  
| [![Florian Eder](https://github.com/FlorianEder.png?size=100)](https://github.com/FlorianEder) | [![John Tran](https://github.com/JohniMIEP.png?size=100)](https://github.com/JohniMIEP) | [![Moritz Enderle](https://github.com/THDMoritzEnderle.png?size=100)](https://github.com/THDMoritzEnderle) |
|:---:|:---:|:---:|
| [Florian Eder](https://github.com/FlorianEder) | [John Tran](https://github.com/JohniMIEP) | [Moritz Enderle](https://github.com/THDMoritzEnderle) |

</div>

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.