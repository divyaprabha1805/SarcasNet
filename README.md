# Sarcasm and HateSpan Detection

Sarcasm and HateSpan Detection is a machine learning project designed to classify text as sarcastic, hate speech, or neutral. The project utilizes deep learning models and NLP techniques to analyze text and highlight specific words that indicate sarcasm or hate speech.

## Features
- **Text Classification**: Detect sarcasm and hate speech in textual data.
- **Highlighted Words**: Identify and highlight words contributing to sarcasm or hate speech.
- **Multiple Models**: Utilizes various models including LSTM, Decision Tree, SVC, KNN, and Naive Bayes.
- **Performance Metrics**: Evaluates models using AUC-ROC and confusion matrices.
- **Streamlit Web App**: User-friendly interface to input text and get predictions.

## Tech Stack
- **Programming Language**: Python
- **Frameworks/Libraries**: TensorFlow, Keras, Scikit-learn, NLTK, Pandas, NumPy
- **Web Interface**: Streamlit

## Installation
### Prerequisites
- Python 3.x installed
- Required libraries installed via pip

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/Praneetha-NM/Sarcasm-and-HateSpan-Detection.git
   cd Sarcasm-and-HateSpan-Detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit web app:
   ```sh
   streamlit run app.py
   ```

## Dataset
- **Sarcasm Detection Dataset**: Contains labeled sarcastic and non-sarcastic text.
- **HateSpan Dataset**: Identifies words contributing to hate speech.

## Usage
- Enter text in the web app.
- Get real-time predictions for sarcasm and hate speech.
- View highlighted words contributing to classification.

## Troubleshooting
- If the model fails to load, ensure all dependencies are installed.
- If Streamlit does not start, try running:
  ```sh
  streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
  ```

## Contact
For any queries, feel free to reach out:
- GitHub: [Praneetha-NM](https://github.com/Praneetha-NM)
- Email: [praneetha7597@gmail.com](mailto:praneetha7597@gmail.com)

