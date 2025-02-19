🔍 Fraud Detection System using Flask
A Flask-based web application that predicts fraudulent transactions using a machine learning model. The model analyzes transaction details and classifies them as Genuine or Fraudulent to enhance financial security.

🚀 Live Demo
🔗 Visit the Website

📌 Features
✅ Fraud Detection – Uses machine learning to classify transactions.
✅ User-Friendly Interface – Simple and interactive web form.
✅ Real-Time Predictions – Instantly displays results.
✅ Secure Deployment – Hosted on Render for public access.

🛠️ Tech Stack
Frontend: HTML, CSS
Backend: Flask (Python)
Machine Learning: scikit-learn, Pandas, NumPy
Deployment: Render

📂 Project Structure
Credit-Card-ML-Project
│── .git
│── .gitignore
│── app.py
│── Procfile
│── README.md
│── requirements.txt
│── setup.py
│── project_structure.txt
│── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── prediction_pipeline.py
│── static/
│   ├── css/
│   │   ├── home.css
│   │   ├── index.css
│── templates/
│   ├── home.html
│   ├── index.html
│── notebook/
│── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── raw.csv
│   ├── test.csv
│   ├── train.csv
│── logs/
│── catboost_info/
│── venv/

💻 Setup & Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/fraud-detection-app.git
cd fraud-detection-app

2️⃣ Create & Activate Virtual Environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Flask App Locally
python app.py
Open http://127.0.0.1:10000 in your browser.

📌 Usage
Enter transaction details in the web form.
Click "Predict".
The system displays:
✅ Genuine Transaction (Safe)
❌ Fraudulent Transaction (Potential Fraud)

🤖 Machine Learning Model
The prediction model was trained using:

Algorithm: Random Forest / XGBoost / Other
Accuracy: 99% (Optional)

📜 License
This project is licensed under the MIT License.

👨‍💻 Author
Sooraj T V
📧 soorajraju9485@gmail.com

🎯 Contributions are welcome! Feel free to open issues or submit pull requests. 🚀

📢 Next Steps
Add unit tests for validation.
Improve UI/UX.


