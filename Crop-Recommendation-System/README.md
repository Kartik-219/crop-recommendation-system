# 🌾 Crop Recommendation System

A machine learning–powered Flask web application that recommends the most suitable crop to grow based on soil nutrients and environmental conditions.

---

## 📌 Features

- Predicts the best crop based on:
  - Nitrogen (N), Phosphorus (P), Potassium (K)
  - Temperature, Humidity
  - pH level, Rainfall
- Trained with real-world agricultural data
- Interactive web interface using HTML/CSS
- Model serialized with pickle (`.pkl`) files

---

## 📂 Project Structure

Crop-Recommendation-System/
├── app.py # Flask backend
├── Crop_Classification_Model.ipynb # Jupyter Notebook used for training
├── data/
│ └── Crop_recommendation.csv # Dataset
├── model/
│ ├── labelencoder.pkl
│ ├── minmaxscaler.pkl
│ ├── model.pkl
│ └── standardscaler.pkl
├── static/
│ └── img.png # UI image
├── templates/
│ └── web.html # Web UI template (Jinja2)
├── .gitignore
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 🚀 How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Kartik-219/crop-recommendation-system.git
cd crop-recommendation-system
2. Create and Activate Virtual Environment
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # macOS/Linux
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Run the Flask App
bash
Copy
Edit
python app.py
Open your browser and go to:
👉 http://127.0.0.1:5000

🧪 Machine Learning Model
Built with scikit-learn

Preprocessing: Label Encoding, Standard Scaling, MinMax Scaling

Model saved as model.pkl

🛠 Tech Stack
Python

Flask

scikit-learn

HTML5 / CSS3

VS Code

👨‍💻 Author
Kartik Kavade
B.Tech in Computer Science (Data Science)
GitHub: Kartik-219

📜 License
This project is licensed under the MIT License

yaml
Copy
Edit

---

### 📌 Next Step:

1. Save this text into a file called `README.md` in your project folder (replace if it exists).
2. Then run:

```bash
git add README.md
git commit -m "Updated README.md with complete details"
git push origin master