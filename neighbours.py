import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

def generate_data(columns, options, num_rows):
    data = {}
    for column in columns:
        if column == "relationship_values":
            data[column] = [', '.join(random.sample(options[column], random.randint(1, len(options[column])))) for _ in range(num_rows)]
        elif column in ["who_are_you", "looking_for", "music_preference", "deal_breakers", "relationship_type"]:
            weights = options.get(column + "_weights", None)
            data[column] = random.choices(options[column], weights=weights, k=num_rows)
        else:
            weights = options.get(column + "_weights", None)
            data[column] = random.choices(options[column], weights=weights, k=num_rows)
    df = pd.DataFrame(data)
    return df

columns = ["who_are_you", "looking_for", "relationship_type", "self_description", "preference", "music_preference", "relationship_values", "deal_breakers"]
options = {
    "who_are_you": ["straight_male", "straight_female", "lesbian", "gay", "bisexual", "transgender", "queer"],
    "who_are_you_weights": [0.6, 0.40, 0.01, 0.01, 0.01, 0.01, 0.01],  
    "looking_for": ["straight_male", "straight_female", "lesbian", "gay", "bisexual", "transgender", "queer"],
    "looking_for_weights": [0.48, 0.47, 0.01, 0.01, 0.01, 0.01, 0.01],  
    "relationship_type": ["Something Casual", "Something Serious"],
    "relationship_type_weights": [0.4, 0.6],  
    "self_description": ["Extrovert", "Introvert", "Somewhere in Between"],
    "self_description_weights": [0.2, 0.2, 0.6],  
    "preference": ["Morning Person", "Night Owl"],
    "preference_weights": [0.25, 0.75], 
    "music_preference": ["Pop", "Rock", "Classical", "Romantic Bollywood", "Hollywood Rizz Songs", "Country"],
    "music_preference_weights": [0.15, 0.15, 0.1, 0.2, 0.1, 0.1],  
    "relationship_values": ["Trust and honesty", "Communication", "Mutual respect", "Shared interests", "Independence", "Emotional support"],
    "deal_breakers": ["Smoking", "Drinking", "Political views", "Religious views"], 
}
num_rows = 1000

df = generate_data(columns, options, num_rows)
df.to_csv("random_data.csv", index=False)

print("Random data generated and saved to 'random_data.csv'.")

le_dict = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    le_dict[column] = le

data_fr = pd.read_csv("random_data.csv")
knn = NearestNeighbors(n_neighbors=10)
knn.fit(df)

distances, indices = knn.kneighbors([df.iloc[0]])

df_inverse = df.copy()
for column in df_inverse.columns:
    df_inverse[column] = le_dict[column].inverse_transform(df_inverse[column])

# Corrected line: df_inverse.iloc[indices[0]] instead of df_inverse.iloc()[indices[0]]
new_df = df_inverse.iloc[indices[0]]

# Create a new DataFrame to store valid entries
valid_entries_df = pd.DataFrame(columns=df_inverse.columns)

# Iterate through the indices and check the condition
for idx in new_df.index:
    entry = new_df.loc[idx]
    if entry["who_are_you"] == data_fr.iloc[0]["looking_for"] and data_fr.iloc[0]["who_are_you"] == entry["looking_for"]:
        valid_entries_df = valid_entries_df.append(entry, ignore_index=True)

# Print or use the new DataFrame with valid entries
help = len(valid_entries_df)
print("Number of valid entries: ", help)

print(valid_entries_df.head())