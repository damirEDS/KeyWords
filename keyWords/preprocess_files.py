import os
import re

directory = "datas"


def remove_punctuation(sentence):
    # Function to remove punctuation from a sentence using regex
    processed_sentence = re.sub(r'[^\w\s]', '', sentence)
    return processed_sentence


# Iterate over files in the specified directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        # Open the file for reading and writing ('r+')
        with open(file_path, 'r+', encoding='utf-8') as file:
            content = file.read()
            # Process the content by converting it to lowercase and removing punctuation
            processed_content = remove_punctuation(content.lower())
            # Move the file pointer to the beginning of the file
            file.seek(0)
            # Write the processed content back to the file
            file.write(processed_content)
            # Truncate the file from the current position to remove any remaining content
            file.truncate()
