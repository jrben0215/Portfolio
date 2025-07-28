from transformers import pipeline
import pandas as pd

file_path = 'C:\\Users\\jrben\\OneDrive\\Desktop\\spam.csv'
spam = pd.read_csv(file_path, encoding='ISO-8859-1')

#selected sample
sample_data = spam.sample(10, random_state=42) if len(spam) >= 10 else spam

#loading a pre-trained zero-shot classification model from Hugging Face
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

#initializing candidate labels
candidate_labels = ["spam", "ham"]

#performing the classification
predictions = []
for text in sample_data['v2']:
    result = classifier(text, candidate_labels)
    predictions.append(result['labels'][0])  # The top prediction

#adding column of predicted class
sample_data['predicted_class'] = predictions

#printing our new dataframe with the column of predicted values added on
print(sample_data[['v1', 'v2', 'predicted_class']])
