# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 2️⃣ Read Dataset
df = pd.read_csv("titanic.csv")

print("First 5 Rows of Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# 3️⃣ Data Cleaning

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing Age with mean
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Drop Cabin column if it exists
if "Cabin" in df.columns:
    df = df.drop("Cabin", axis=1)

# Fill missing Embarked with mode
if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Remove duplicates
df = df.drop_duplicates()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# 4️⃣ Data Visualization

# 📊 1. Survival Count
plt.figure()
survival_counts = df["Survived"].value_counts()
plt.bar(["Did Not Survive", "Survived"], survival_counts)
plt.title("Survival Count")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")
plt.show()


# 📊 2. Age Distribution
plt.figure()
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# 📊 3. Passenger Class Distribution
plt.figure()
class_counts = df["Pclass"].value_counts()
plt.bar(class_counts.index, class_counts.values)
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()


print("\nAnalysis Completed Successfully ✅")
