# 📚 Learning Path Recommendation System

A personalized recommendation system that suggests the **next best learning topics/courses** for intern based on their past learning history.
This project uses **Collaborative Filtering (SVD)** and **Flask** to provide recommendations in a simple web interface.

---

## 🚀 Features

* 📖 Recommends the next course/topic based on user history
* 🧠 Uses **Machine Learning (Surprise SVD)** for personalized recommendations
* 🌐 Flask-powered web application
* 🗂️ Organized project structure

---

## 🗂️ Project Structure

```
Learning-Path-Recommendation-System/
│
├── models/                  # Trained models (.pkl files)
├── templates/               # HTML templates for Flask
│   └── index.html
│
├── app.py                   # Flask web app
├── model_trainer.py         # Script to train and save the model
├── recommender.py           # Loads model & generates recommendations
├── learning_path_dataset.csv # Dataset (student-course interactions)
├── README.md                # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/HadiaWaheed/Learning-Path-Recommendation-System.git
cd Learning-Path-Recommendation-System
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The dataset (`learning_path_dataset.csv`) contains user-course interactions.
Columns include:

* `Intern_ID` → Unique learner ID
* `Course_ID / Topic` → Course or topic identifier
* `Rating` → User rating (1–5 scale)
* `Completed` → Whether the course was completed

---

## 🧠 Model Training

To train and save the recommendation model:

```bash
python model_trainer.py
```

This will save:

* `models/recommendation_model.pkl` → Trained ML model
* `models/course_map.pkl` → Mapping of course details

---

## 🔎 Running the App

Start the Flask web server:

```bash
python app.py
```

Then open 👉 `http://127.0.0.1:5000/` in your browser.

---

## 🖥️ Example Usage

* Enter an **Intern ID** (e.g., `I24`)
* System will show:

  * ✅ Past completed topics
  * 🎯 New recommended learning paths

---

## 📦 Dependencies

* Python 3.8+
* pandas
* scikit-surprise
* Flask

Install manually if needed:

```bash
pip install pandas scikit-surprise flask
```

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👩‍💻 Author

**Hadia Waheed**
