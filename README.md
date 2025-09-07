# 🧠 MoEffNet Breast Cancer Classification App

A professional Streamlit application for breast cancer patch classification using Multi-Expert EfficientNet (MoEffNet) with interactive mammogram analysis.

## 🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## ✨ Features

- **Interactive Mammogram Analysis**: Click and drag to select regions of interest
- **Multi-Expert AI**: 4 specialized neural networks with intelligent gating
- **Explainable AI**: GradCAM and attention visualization
- **Professional UI**: Medical-grade interface with confidence scoring
- **Real-time Classification**: Instant analysis of selected regions

## 🏗️ Architecture

**MoEffNet (Multi-Expert EfficientNet)** combines:
- EfficientNet-B1 backbone (ImageNet pretrained)
- 4 specialized expert networks
- Intelligent gating network
- Confidence estimation system

## 📊 Classes
- **Normal**: Healthy tissue
- **Benign**: Non-cancerous abnormalities  
- **Malignant**: Potentially cancerous tissue

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/moeffnet-breast-cancer-app.git
cd moeffnet-breast-cancer-app
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run BreastCancer_MoEffNet_Streamlit.py
```

## 🤗 Model Details

The trained MoEffNet model is hosted on Hugging Face:
- **Repository**: [your-username/moeffnet-breast-cancer](https://huggingface.co/your-username/moeffnet-breast-cancer)
- **Model Size**: ~85MB
- **Architecture**: Multi-Expert EfficientNet-B1
- **Input Size**: 224×224 RGB images

## 📱 Usage

1. **Upload** a mammogram image (PNG, JPEG, or URL)
2. **Draw a rectangle** around suspicious areas
3. **Click "Analyze Region"** to get AI classification
4. **View detailed results** with expert analysis and XAI

## 🔬 Technical Details

- **Framework**: PyTorch + Streamlit
- **Backbone**: EfficientNet-B1
- **Experts**: 4 specialized networks
- **XAI**: GradCAM + Feature attention
- **Deployment**: Streamlit Cloud + Hugging Face

## ⚠️ Disclaimer

This application is for **research and educational purposes only**. It is not intended for clinical diagnosis. Always consult qualified healthcare professionals for medical advice.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- Streamlit for the amazing framework
- Hugging Face for model hosting
