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
def clean_data(name):
    df = pd.read_csv(name)
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    columns_for_training = ['Email Address','Who are you?','Who are you looking for?','Are you looking for:', 'Would you describe yourself more as an introvert or an extrovert?', 'Are you a morning person or a night owl?','What type of music do you prefer?','Which of these values is most important to you in a relationship?','Are there any absolute deal breakers for you in a relationship?']
    df = df[columns_for_training]
    return df

def ml_funct(df,file_name):
    df.to_csv("file_name", index=False)
    le_dict = {}
    for column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        le_dict[column] = le

    data_fr = pd.read_csv("file_name")
    knn = NearestNeighbors(n_neighbors=100)
    knn.fit(df)

    distances, indices = knn.kneighbors([df.iloc[0]])

    df_inverse = df.copy()
    for column in df_inverse.columns:
        df_inverse[column] = le_dict[column].inverse_transform(df_inverse[column])

    new_df = df_inverse.iloc[indices[0]]

    valid_entries_df = pd.DataFrame(columns=df_inverse.columns)

    for idx in new_df.index:
        entry = new_df.loc[idx]
        if entry["Who are you?"] == data_fr.iloc[0]["Who are you looking for?"] and data_fr.iloc[0]["Who are you?"] == entry["Who are you looking for?"]:
            valid_entries_df = pd.concat([valid_entries_df, entry.to_frame().T], ignore_index=True)
    return valid_entries_df

num_rows = 1000

df = clean_data("ML_blind_date.csv")
valid_entries_df = ml_funct(df,"lol")

help = len(valid_entries_df)
print("Number of valid entries: ", help)

valid_entries_df.to_csv("lol.csv",index = True)