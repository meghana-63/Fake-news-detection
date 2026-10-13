import pandas as pd

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

print("Fake news articles:", len(fake))
print("Real news articles:", len(true))

print("\nFake dataset columns:")
print(fake.columns.tolist())

print("\nReal dataset columns:")
print(true.columns.tolist())

print("\nFake dataset:")
print(fake.head())

print("\nReal dataset:")
print(true.head())