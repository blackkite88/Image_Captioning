
  # 📸 Image Captioning using VGG16 + LSTM  
  A deep learning project trained on the **Flickr8k** dataset ✨

  This project implements an end-to-end image captioning model using a **VGG16 CNN encoder**
  and an **LSTM decoder**. The full workflow is implemented in `project.ipynb`.

  ---

  ## 🚀 Project Overview  
  **🧠 Encoder (CNN): VGG16**
  - Uses pretrained VGG16 weights  
  - Extracts a **4096-dimensional** feature vector  
  - Features are passed through **Dense(256, relu)**  

  **📝 Decoder (RNN): LSTM**
  - Embedding(256)  
  - LSTM(256 units)  
  - Combined with CNN output via `add()`  
  - Dense(256, relu)  
  - Dense(vocab_size, softmax)  

  This encoder–decoder architecture allows the model to generate meaningful captions
  based on the extracted image features.

  ---

  ## 🗂 Dataset Structure (Flickr8k)

      Image_Captioning/
      │
      ├── project.ipynb
      ├── images/                    # 📁 Flickr8k images here
      └── captions/
          └── Flickr8k.token.txt     # 📝 Caption file

  You must download the dataset manually (e.g., via Kaggle).

  ---

  ## 🔧 Installation  
  ```bash
# Clone repository
git clone https://github.com/blackkite88/Image_Captioning.git
cd Image_Captioning

# (Optional) create virtual environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

# Install dependencies
pip install tensorflow numpy pandas matplotlib pillow nltk tqdm jupyter scikit-learn

# Launch notebook
jupyter notebook project.ipynb

```

  ---

  ## ▶️ Running the Notebook  
  Use Jupyter Notebook to open and run:

  - Load & clean captions 🧹  
  - Tokenize & build vocabulary 🔤  
  - Extract VGG16 image features 🖼️  
  - Prepare sequences  
  - Train the model 🏋️  
  - Generate captions 🗣️  

  ---

  ## 🧱 Model Architecture (Exact)

      Encoder:
        VGG16 → 4096-dim vector → Dense(256, relu)

      Decoder:
        Embedding(256) → LSTM(256) → Add() → Dense(256, relu) → Dense(vocab_size, softmax)

  ---

  ## 🌟 Future Improvements  
  - Add attention mechanism 🎯  
  - Modern CNN encoders (ResNet, EfficientNet, Inception) 🏗  
  - Build a Gradio or Streamlit app 🌐  
  - Convert notebook into modular scripts 🧩  

  ---


