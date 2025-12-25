# 😷 Face Mask Detection - End-to-End ML Pipeline


A complete end-to-end machine learning pipeline for detecting face masks in images using deep learning.

## 🎯 Features

- **Multi-face Detection**: Detects multiple faces in a single image using MTCNN
- **3-Class Classification**: With mask, without mask, mask worn incorrectly
- **Complete Pipeline**: Train → Evaluate → Deploy workflow
- **CI/CD Ready**: GitHub Actions for automated training and testing
- **Streamlit Web App**: Interactive UI for testing the model

## 📁 Project Structure

```
├── .github/workflows/    
│   ├── ml-pipeline.yml   
│   └── code-quality.yml  
├── src/
│   ├── config.py          
│   ├── dataset.py         
│   └── model.py          
├── tests/                 
├── data/
│   ├── images/            
│   └── annotations/       
├── pipeline.py          
├── evaluate.py        
├── app.py                 
└── requirements.txt      
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python pipeline.py
```

### 3. Evaluate
```bash
python evaluate.py
```

### 4. Run the Web App
```bash
streamlit run app.py
```

## 📊 Dataset

- **Format**: Images with Pascal VOC XML annotations
- **Classes**:
  - `with_mask` (0)
  - `without_mask` (1)
  - `mask_weared_incorrect` (2)

## ⚙️ Configuration

Edit `src/config.py` to customize:


## 🔄 CI/CD Pipeline

The GitHub Actions pipeline automatically:

1. **Lint & Test** - Code quality checks
2. **Train** - Trains the model with configurable epochs
3. **Evaluate** - Generates metrics and confusion matrix
4. **Deploy Check** - Validates deployment readiness

### Manual Trigger
Go to Actions → ML Pipeline → Run workflow (can specify epochs/batch_size)

## 📈 Model Architecture

- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Output Heads**:
  - Bounding box regression (4 units, sigmoid)
  - Classification (3 units, softmax)
- **Loss**: MSE (bbox) + Categorical Crossentropy (class)

## 📝 License

MIT License

## Made with love by Harivansh Bhardwaj❤️