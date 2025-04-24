# CS 5330 Final Project 
# Shihua Lu, Chuhan Ren
# This is used to plot confusion matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load predictions
df = pd.read_csv("10_predictions.csv")

# Extract true label from filename
df['true_label'] = df['filename'].apply(lambda x: x.split('_')[0])
df['pred_label'] = df['predicted_class']

# Define full list of expected labels
expected_labels = ['buffalo', 'cheetah', 'elephant', 'fox', 'hyena',
                   'lion', 'rhino', 'tiger', 'wolf', 'zebra']

# Check for any missing labels
found_labels = sorted(df['true_label'].unique())
print(" Found true labels in dataset:", found_labels)
missing = set(expected_labels) - set(found_labels)
if missing:
    print(f" Warning: Missing expected labels from dataset: {missing}")

# Create confusion matrix with full class list
cm = confusion_matrix(df['true_label'], df['pred_label'], labels=expected_labels)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=expected_labels, yticklabels=expected_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - ResNet18 (10 Classes)")
plt.tight_layout()
plt.savefig("confusion_matrix_resnet18_10class_fixed.png")
plt.show()