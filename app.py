# app.py
from flask import Flask, render_template, request
import os, pickle
import pandas as pd

app = Flask(__name__, template_folder="templates")

# Paths
MODEL_PATH = os.path.join("models", "recommendation_model.pkl")
MAP_PATH = os.path.join("models", "course_map.pkl")
DATA_PATH = "learning_path_dataset.csv"

# Load Surprise model (if available) or fallback to simple model
model = None
course_map = {}

# --- Fallback simple recommender (pure Python) ---
class SimpleRecommender:
    def __init__(self):
        self.global_mean = 3.5
        self.intern_means = {}
        self.topic_means = {}
        self.df = None

    def fit(self, df):
        self.df = df
        if len(df) > 0:
            self.global_mean = df['Rating'].mean()
            self.intern_means = df.groupby('InternID')['Rating'].mean().to_dict()
            self.topic_means = df.groupby('Topic')['Rating'].mean().to_dict()

    def predict(self, intern_id, topic):
        im = self.intern_means.get(intern_id, self.global_mean)
        tm = self.topic_means.get(topic, self.global_mean)
        return max(1.0, min(5.0, 0.6 * im + 0.4 * tm))

# Try loading Surprise model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Loaded Surprise model from", MODEL_PATH)
except Exception as e:
    print("Could not load Surprise model:", e)
    model = None

# Load course_map (topic -> {link, category, difficulty})
try:
    with open(MAP_PATH, "rb") as f:
        course_map = pickle.load(f)
    print("Loaded course_map from", MAP_PATH)
except Exception as e:
    print("Could not load course_map:", e)
    course_map = {}

# Load dataset for past topics (if available)
if os.path.exists(DATA_PATH):
    df_all = pd.read_csv(DATA_PATH)
    # Ensure columns exist and standard names
    expected_cols = ['InternID','Topic','Link','Rating','Completed','Category','Difficulty']
    # no strict enforcement; we'll use what is present
    if 'Completed' not in df_all.columns:
        df_all['Completed'] = 1
else:
    df_all = pd.DataFrame(columns=['InternID','Topic','Link','Rating','Completed','Category','Difficulty'])

# If Surprise model not available, train SimpleRecommender on df_all
if model is None:
    print("Training fallback SimpleRecommender...")
    fallback = SimpleRecommender()
    filtered = df_all[(df_all['Completed'] == 1) & (df_all['Rating'] > 0)] if not df_all.empty else df_all
    fallback.fit(filtered)
else:
    fallback = None

# Universal prediction util
def predict_rating(intern_id, topic):
    # Use Surprise model if it appears to have predict()
    try:
        if model is not None and hasattr(model, "predict") and callable(model.predict):
            # Surprise predict expects (user, item)
            pred = model.predict(intern_id, topic)
            # Surprise Predict object may have .est
            return float(getattr(pred, "est", pred))
    except Exception as e:
        print("Surprise predict error:", e)

    # fallback
    return float(fallback.predict(intern_id, topic))

@app.route("/", methods=["GET", "POST"])
def index():
    intern_id = ""
    past_topics = []
    recommendations = []
    error_message = ""

    if request.method == "POST":
        intern_id = request.form.get("intern_id", "").strip()
        if intern_id == "":
            error_message = "Please enter an Intern ID."
            return render_template("index.html", intern_id=intern_id,
                                   past_topics=past_topics,
                                   recommendations=recommendations,
                                   error_message=error_message)

        # Check intern presence
        if intern_id not in df_all['InternID'].values:
            error_message = f"Intern '{intern_id}' not found in dataset."
            return render_template("index.html", intern_id=intern_id,
                                   past_topics=past_topics,
                                   recommendations=recommendations,
                                   error_message=error_message)

        # Past topics (completed entries)
        intern_rows = df_all[(df_all['InternID'] == intern_id) & (df_all['Completed'] == 1)].copy()
        # Keep original order if you want: intern_rows.sort_values(...)
        for _, r in intern_rows.iterrows():
            topic = r.get('Topic', '')
            past_topics.append({
                'topic': topic,
                'rating': r.get('Rating', ''),
                'link': r.get('Link', course_map.get(topic, {}).get('link', '')),
                'category': r.get('Category', course_map.get(topic, {}).get('category', '')),
                'difficulty': r.get('Difficulty', course_map.get(topic, {}).get('difficulty', ''))
            })

        # Candidate topics = all topics in df_all minus those already taken
        all_topics = df_all['Topic'].unique().tolist()
        taken = intern_rows['Topic'].unique().tolist()
        candidates = [t for t in all_topics if t not in taken]

        # Predict ratings for candidates
        preds = []
        for t in candidates:
            est = predict_rating(intern_id, t)
            link = course_map.get(t, {}).get('link', '')
            category = course_map.get(t, {}).get('category', '')
            preds.append((t, round(float(est),2), link, category))

        preds.sort(key=lambda x: x[1], reverse=True)

        # Build a step-by-step path (top 6)
        for i, (topic, est, link, category) in enumerate(preds[:6], start=1):
            recommendations.append({
                'step': i,
                'topic': topic,
                'predicted_rating': est,
                'link': link,
                'category': category
            })

    return render_template("index.html",
                           intern_id=intern_id,
                           past_topics=past_topics,
                           recommendations=recommendations,
                           error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
