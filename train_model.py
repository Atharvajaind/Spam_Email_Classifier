import pandas as pd

# -------------------------------
# STEP 1 : Load Dataset
# -------------------------------

print("=" * 60)
print("        SPAM EMAIL CLASSIFIER")
print("=" * 60)

df = pd.read_csv(
    "dataset/spam.tsv",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("\n✅ Dataset Loaded Successfully")

# -------------------------------
# STEP 2 : Dataset Information
# -------------------------------

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nSpam/Ham Count:")
print(df["label"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# STEP 3 : Remove Duplicate Rows
# -------------------------------

print("\nRemoving Duplicate Messages...")

df = df.drop_duplicates()

print("Dataset Shape After Removing Duplicates:")
print(df.shape)

# -------------------------------
# STEP 4 : Convert Labels
# -------------------------------

print("\nConverting Labels...")

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print(df.head())

print("\nLabel Conversion Completed!")
print("\nFinal Dataset Information")

print(df.info())

print("\n✅ Data Preprocessing Completed Successfully!")


# STEP 5 : TF-IDF Feature Engineering

from sklearn.feature_extraction.text import TfidfVectorizer

# Features (SMS messages)
X = df["message"]

# Target (Spam/Ham)
y = df["label"]

print("\nCreating TF-IDF Vectorizer...")

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

print("✅ TF-IDF Vectorization Completed!")

print("\nShape of Feature Matrix:")
print(X.shape)


# STEP 6 : Train-Test Split

from sklearn.model_selection import train_test_split

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("✅ Dataset Split Successfully!")

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# STEP 7 : Train Naive Bayes Model

from sklearn.naive_bayes import MultinomialNB

print("\nTraining the Machine Learning Model...")

model = MultinomialNB()

model.fit(X_train, y_train)

print("✅ Model Trained Successfully!")


# STEP 8 : Make Predictions


print("\nMaking Predictions on Test Data...")

y_pred = model.predict(X_test)

print("✅ Prediction Completed!")


# STEP 9 : Model Evaluation


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# STEP 10 : Save the Trained Model


import joblib

print("\nSaving Machine Learning Model...")

joblib.dump(model, "models/spam_classifier.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Model Saved Successfully!")
print("📁 Files saved inside the models folder.")