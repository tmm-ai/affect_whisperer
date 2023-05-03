import pandas as pd

# fer2013_file_path = "icml_face_data.csv"
# ferplus_file_path = "fer2013new.csv"

# Read the FER2013 and FER+ data
fer2013_data = pd.read_csv("icml_face_data.csv")
ferplus_labels = pd.read_csv("fer2013new.csv")

# Drop the 'emotion' column from FER2013 data, as we'll use the FER+ labels
fer2013_data = fer2013_data.drop(columns=['emotion'])

# Merge the data using index
merged_data = fer2013_data.merge(ferplus_labels, left_index=True, right_index=True)

# Save the merged data to a new CSV file
merged_data.to_csv("merged_fer_data.csv", index=False)
