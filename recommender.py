# recommender.py
import pandas as pd
import numpy as np
from collections import defaultdict

class SimpleRecommender:
    """
    A simple recommendation system based on collaborative filtering
    """
    def __init__(self):
        self.intern_means = {}
        self.course_means = {}
        self.global_mean = 0
        self.interactions_df = None
    
    def fit(self, interactions_df):
        """Train the model"""
        self.interactions_df = interactions_df
        
        # Calculate global mean rating
        self.global_mean = interactions_df['Rating'].mean()
        
        # Calculate mean rating for each intern
        self.intern_means = interactions_df.groupby('Intern_ID')['Rating'].mean().to_dict()
        
        # Calculate mean rating for each course
        self.course_means = interactions_df.groupby('Course_ID')['Rating'].mean().to_dict()
    
    def predict(self, intern_id, course_id):
        """Predict rating for a given intern and course"""
        # If we have no data, return global mean
        if intern_id not in self.intern_means or course_id not in self.course_means:
            return self.global_mean
        
        # Simple prediction: average of intern mean and course mean
        intern_mean = self.intern_means.get(intern_id, self.global_mean)
        course_mean = self.course_means.get(course_id, self.global_mean)
        
        # Weighted average
        prediction = 0.6 * intern_mean + 0.4 * course_mean
        
        # Ensure prediction is within valid range
        return max(1, min(5, prediction))
    
    def recommend(self, intern_id, n=5):
        """Get top n recommendations for an intern"""
        # Get all courses
        all_courses = self.interactions_df['Course_ID'].unique()
        
        # Get courses already taken by this intern
        taken_courses = self.interactions_df[
            self.interactions_df['Intern_ID'] == intern_id
        ]['Course_ID'].values
        
        # Predict ratings for courses not taken
        recommendations = []
        for course_id in all_courses:
            if course_id not in taken_courses:
                pred_rating = self.predict(intern_id, course_id)
                recommendations.append((course_id, pred_rating))
        
        # Sort by predicted rating (descending) and return top n
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]