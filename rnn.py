import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk

# Download NLTK data files
nltk.download('punkt')

# Load South Indian datasets
south_temple = pd.read_excel(r'D:\deeplearning project\culture_heritage_app\dataset\south\South_Indian_temple.xlsx')
south_dance = pd.read_excel(r'D:\deeplearning project\culture_heritage_app\dataset\south\South_Indian_dance.xlsx')
south_food = pd.read_excel(r'D:\deeplearning project\culture_heritage_app\dataset\south\South_Indian_food.xlsx')
south_music = pd.read_excel(r'D:\deeplearning project\culture_heritage_app\dataset\south\South_Indian_music.xlsx')

# Load North Indian datasets
north_temple = pd.read_excel(r'D:\deeplearning project\culture_heritage_app\dataset\north\North_Indian_temple.xlsx')
north_food = pd.read_excel(r'D:\deeplearning project\culture_heritage_app\dataset\north\North_Indian_food.xlsx')
north_dance = pd.read_excel(r'D:\deeplearning project\culture_heritage_app\dataset\north\North_Indian_dance.xlsx')
north_music = pd.read_excel(r'D:\deeplearning project\culture_heritage_app\dataset\north\North_Indian_music.xlsx')

# Process the 'Notes/Beliefs' and 'Description' columns
def process_data(df, column):
    if column in df.columns:
        df['Processed_Description'] = df[column].str.lower()
        return df[['Processed_Description']]
    else:
        print(f"Column '{column}' not found in dataset.")
        return pd.DataFrame()

# Process South Indian datasets
south_temple = process_data(south_temple, 'Notes/Beliefs')
south_dance = process_data(south_dance, 'Description')
south_food = process_data(south_food, 'Description')
south_music = process_data(south_music, 'Description')

# Process North Indian datasets
north_temple = process_data(north_temple, 'Notes/Beliefs')
north_food = process_data(north_food, 'Description')
north_dance = process_data(north_dance, 'Description')
north_music = process_data(north_music, 'Description')

# Combine all datasets
all_data = pd.concat([
    south_temple.assign(label='south_temple'),
    south_dance.assign(label='south_dance'),
    south_food.assign(label='south_food'),
    south_music.assign(label='south_music'),
    north_temple.assign(label='north_temple'),
    north_food.assign(label='north_food'),
    north_dance.assign(label='north_dance'),
    north_music.assign(label='north_music')
], ignore_index=True)

# Tokenizer and sequence processing
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(all_data['Processed_Description'])
sequences = tokenizer.texts_to_sequences(all_data['Processed_Description'])
padded_sequences = pad_sequences(sequences, maxlen=100)

# Convert labels to integers
label_mapping = {label: index for index, label in enumerate(all_data['label'].unique())}
encoded_labels = np.array([label_mapping[label] for label in all_data['label']])

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Load pre-trained GloVe embeddings
embedding_index = {}
with open(r'C:\Users\Ushadevi M\Downloads\glove.6B\glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Create embedding matrix
embedding_matrix = np.zeros((10000, 100))
for word, i in tokenizer.word_index.items():
    if i < 10000:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Build Bi-LSTM model with pre-trained embeddings and additional techniques
model = Sequential([
    Embedding(input_dim=10000, output_dim=100, input_length=100, weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(units=128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(units=64)),
    Dropout(0.5),
    Dense(units=128, activation='relu'),
    Dropout(0.5),
    Dense(units=len(label_mapping), activation='softmax')
])

# Compile the model with a different optimizer
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with more epochs
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model and tokenizer
model.save('rnn_model.keras')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
