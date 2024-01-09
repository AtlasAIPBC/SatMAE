# min
import os
import json
import pandas as pd


label_mapping = {
    "original_labels":{
        "Continuous urban fabric": 0,
        "Discontinuous urban fabric": 1,
        "Industrial or commercial units": 2,
        "Road and rail networks and associated land": 3,
        "Port areas": 4,
        "Airports": 5,
        "Mineral extraction sites": 6,
        "Dump sites": 7,
        "Construction sites": 8,
        "Green urban areas": 9,
        "Sport and leisure facilities": 10,
        "Non-irrigated arable land": 11,
        "Permanently irrigated land": 12,
        "Rice fields": 13,
        "Vineyards": 14,
        "Fruit trees and berry plantations": 15,
        "Olive groves": 16,
        "Pastures": 17,
        "Annual crops associated with permanent crops": 18,
        "Complex cultivation patterns": 19,
        "Land principally occupied by agriculture, with significant areas of natural vegetation": 20,
        "Agro-forestry areas": 21,
        "Broad-leaved forest": 22,
        "Coniferous forest": 23,
        "Mixed forest": 24,
        "Natural grassland": 25,
        "Moors and heathland": 26,
        "Sclerophyllous vegetation": 27,
        "Transitional woodland/shrub": 28,
        "Beaches, dunes, sands": 29,
        "Bare rock": 30,
        "Sparsely vegetated areas": 31,
        "Burnt areas": 32,
        "Inland marshes": 33,
        "Peatbogs": 34,
        "Salt marshes": 35,
        "Salines": 36,
        "Intertidal flats": 37,
        "Water courses": 38,
        "Water bodies": 39,
        "Coastal lagoons": 40,
        "Estuaries": 41,
        "Sea and ocean": 42
    },
    "label_conversion": [
        [0, 1], 
        [2], 
        [11, 12, 13], 
        [14, 15, 16, 18], 
        [17],
        [19], 
        [20], 
        [21], 
        [22], 
        [23], 
        [24],
        [25, 31],
        [26, 27], 
        [28], 
        [29], 
        [33, 34], 
        [35, 36], 
        [38, 39], 
        [40, 41, 42]
    ],
    "BigEarthNet-19_labels":{
        "Urban fabric": 0,
        "Industrial or commercial units": 1,
        "Arable land": 2,
        "Permanent crops": 3,
        "Pastures": 4,
        "Complex cultivation patterns": 5,
        "Land principally occupied by agriculture, with significant areas of natural vegetation": 6,
        "Agro-forestry areas": 7,
        "Broad-leaved forest": 8,
        "Coniferous forest": 9,
        "Mixed forest": 10,
        "Natural grassland and sparsely vegetated areas": 11,
        "Moors, heathland and sclerophyllous vegetation": 12,
        "Transitional woodland, shrub": 13,
        "Beaches, dunes, sands": 14,
        "Inland wetlands": 15,
        "Coastal wetlands": 16,
        "Inland waters": 17,
        "Marine waters": 18
    }
}


# Function to extract information from JSON files
def extract_info_from_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        labels = data.get('labels')
        acquisition_date = data.get('acquisition_date')
    return labels, acquisition_date

# Define the main folder path
main_folder = '/home/ada/satmae/other_data/bigearthnet/BigEarthNet-v1.0'



def map_labels(labels, label_mapping):
    # Flatten label conversion mapping
    flat_conversion = [item for sublist in label_mapping['label_conversion'] for item in sublist]
    
    # Map original labels to BigEarthNet-19 labels
    mapped_labels = []
    for orig_label in labels:
        for idx, conv_labels in enumerate(label_mapping['label_conversion']):
            if label_mapping['original_labels'][orig_label] in conv_labels:
                mapped_labels.append(list(label_mapping['BigEarthNet-19_labels'].keys())[idx])
                break
    
    return mapped_labels

# Initialize an empty DataFrame to store the data
result_df = pd.DataFrame(columns=['category', 'patch_path', 'location_id', 'timestamp'])

# Loop through each subfolder
allowed_subfolders = pd.read_csv('BigEarthNet-S2_19-classes_models-master/splits/train.csv').iloc[:, 0].values.tolist()

# n_subfolders = 20
for root, dirs, files in os.walk(main_folder): 
    folder_name = os.path.basename(root)  # Get the current folder name
    if folder_name in allowed_subfolders:
        json_files = [file for file in files if file.endswith('.json')]

        # Process JSON files in the subfolder
        for json_file in json_files:
            json_path = os.path.join(root, json_file)
            labels, acquisition_date = extract_info_from_json(json_path)

            # Map the labels to BigEarthNet-19 labels
            mapped_labels = map_labels(labels, label_mapping)

            result_df = result_df.append(
                {
                     'category': mapped_labels, 
                     'patch_path': root,  # Save folder path instead of individual file path
                     'timestamp': acquisition_date,
                     'location_id': folder_name
                }, ignore_index=True)

#         n_subfolders -= 1
#         if n_subfolders == 0:
#             break
        
# Export the resulting DataFrame to a CSV file
print(result_df.head())
result_df.to_csv('train_multi_band.csv', index=False)
