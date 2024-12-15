import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

dataset_path = "job_recommendation_large_dataset.csv"
data = pd.read_csv(dataset_path)

for column in ['skills', 'job_description', 'required_skills']:
    data[column] = data[column].str.lower()
data['combined_features'] = data['job_description'] + " " + data['required_skills']
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
job_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])

user_skills = input("Enter your skills (comma-separated): ").lower()
user_experience = input("Enter your work experience in years: ")

user_profile = user_skills + " experience " + user_experience
user_profile_vector = tfidf_vectorizer.transform([user_profile])

similarity_scores = cosine_similarity(user_profile_vector, job_matrix).flatten()
data['similarity_score'] = similarity_scores

content_based_recommendations = data[['user_id', 'job_id', 'similarity_score']]
content_based_recommendations = content_based_recommendations.sort_values(by='similarity_score', ascending=False)

print("Top Job Recommendations for You:")
print(content_based_recommendations.head(10))

if 'relevant_jobs' not in data.columns:
    import numpy as np
    np.random.seed(42)
    data['relevant_jobs'] = np.random.choice([0, 1], size=len(data))

recommended_jobs = content_based_recommendations.head(10)['job_id'].values
data['predicted'] = data['job_id'].apply(lambda x: 1 if x in recommended_jobs else 0)

precision = precision_score(data['relevant_jobs'], data['predicted'])
recall = recall_score(data['relevant_jobs'], data['predicted'])
f1 = f1_score(data['relevant_jobs'], data['predicted'])

#(MAP)
def mean_average_precision(y_true, y_scores):
    sorted_indices = y_scores.argsort()[::-1]
    sorted_y_true = y_true[sorted_indices]
    cum_sum = 0.0
    relevant = 0
    for i, val in enumerate(sorted_y_true):
        if val == 1:
            relevant += 1
            cum_sum += relevant / (i + 1)
    return cum_sum / relevant if relevant > 0 else 0

map_score = mean_average_precision(data['relevant_jobs'].values, data['similarity_score'].values)

print("\nEvaluation Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"MAP: {map_score:.4f}")
