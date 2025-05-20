import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

CSV_FILE = 'data.csv'

data = []
labels = []

# Read data from CSV file
with open(CSV_FILE, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip the header
    for row in csvreader:
        data.append([float(value) for value in row[:-1]])  # Convert features to float
        labels.append(row[-1])  # Last column is the label

data = np.asarray(data)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model using joblib
joblib.dump(model, 'model.joblib')