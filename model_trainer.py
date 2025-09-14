# model_trainer.py
import pandas as pd
import numpy as np
import pickle
import os
from recommender import SimpleRecommender  # Import from the separate file

def generate_sample_data():
    """
    Generate sample data if the actual CSV files are not available.
    This creates realistic intern-course interaction data.
    """
    # Create sample intern-course interactions
    num_interns = 100
    num_courses = 20
    num_interactions = 800  # Each intern has taken ~8 courses on average
    
    intern_ids = [f'I{i+1}' for i in range(num_interns)]
    course_ids = [f'C{i+1}' for i in range(num_courses)]
    
    # Create course name mapping
    course_name_map = {f'C{i+1}': f'Course_{i+1}' for i in range(num_courses)}
    
    # Generate interactions with some pattern
    np.random.seed(42)  # For reproducible results
    
    interactions = []
    for intern_idx in range(num_interns):
        # Each intern takes 5-12 courses
        num_courses_taken = np.random.randint(5, 13)
        courses_taken = np.random.choice(course_ids, num_courses_taken, replace=False)
        
        for course_id in courses_taken:
            # Base rating depends on intern and course "affinity"
            base_rating = 3 + 0.1 * intern_idx % 10 + 0.1 * int(course_id[1:]) % 10
            # Add some randomness
            rating = max(1, min(5, round(base_rating + np.random.normal(0, 0.7), 1)))
            interactions.append({
                'Intern_ID': f'I{intern_idx+1}',
                'Course_ID': course_id,
                'Rating': rating
            })
    
    interactions_df = pd.DataFrame(interactions)
    
    # Create course metadata
    metadata = []
    for i, course_id in enumerate(course_ids):
        metadata.append({
            'Course_ID': course_id,
            'Category': np.random.choice(['Technical', 'Soft Skills', 'Leadership', 'Tools']),
            'Difficulty': np.random.choice(['Beginner', 'Intermediate', 'Advanced']),
            'Duration_hours': np.random.randint(10, 41)
        })
    
    metadata_df = pd.DataFrame(metadata)
    
    return interactions_df, metadata_df, course_name_map

def train_recommendation_model():
    """
    Train the recommendation model using our simple recommender.
    """
    print("Training recommendation model...")
    
    # Check if data files exist, otherwise generate sample data
    if not os.path.exists('intern_course_interactions.csv'):
        print("No interaction data found. Generating sample data...")
        interactions_df, metadata_df, course_name_map = generate_sample_data()
        
        # Save the generated data
        interactions_df.to_csv('intern_course_interactions.csv', index=False)
        metadata_df.to_csv('course_metadata.csv', index=False)
        
        # Save course name mapping
        if not os.path.exists('models'):
            os.makedirs('models')
        with open('models/course_map.pkl', 'wb') as f:
            pickle.dump(course_name_map, f)
    else:
        # Load existing data
        interactions_df = pd.read_csv('intern_course_interactions.csv')
        metadata_df = pd.read_csv('course_metadata.csv')
        
        # Create course name mapping from metadata or generate if not available
        if 'Course_Name' in metadata_df.columns:
            course_name_map = dict(zip(metadata_df['Course_ID'], metadata_df['Course_Name']))
        else:
            # Extract course IDs from interactions
            course_ids = interactions_df['Course_ID'].unique()
            course_name_map = {cid: f'Course_{cid[1:]}' for cid in course_ids}
        
        # Save course name mapping
        if not os.path.exists('models'):
            os.makedirs('models')
        with open('models/course_map.pkl', 'wb') as f:
            pickle.dump(course_name_map, f)
    
    # Train our simple recommender
    model = SimpleRecommender()
    model.fit(interactions_df)
    
    # Test the model with a sample prediction
    sample_intern = interactions_df['Intern_ID'].iloc[0]
    sample_course = interactions_df['Course_ID'].iloc[0]
    sample_pred = model.predict(sample_intern, sample_course)
    
    print(f"Model trained successfully! Sample prediction: {sample_pred:.2f}")
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    with open('models/recommendation_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved to models/recommendation_model.pkl")
    
    return model

if __name__ == '__main__':
    train_recommendation_model()