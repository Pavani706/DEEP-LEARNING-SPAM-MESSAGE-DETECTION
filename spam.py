import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
import os

# Step 1: Load and preprocess the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Encode labels: ham=0, spam=1
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)

# Padding
maxlen = 100
x_train_pad = pad_sequences(x_train_seq, maxlen=maxlen)
x_test_pad = pad_sequences(x_test_seq, maxlen=maxlen)

# Step 2: Build and train the model
model_file = "spam_model.h5"
tokenizer_file = "tokenizer.pkl"

if not os.path.exists(model_file):
    model = Sequential([
        Embedding(10000, 64, input_length=maxlen),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_pad, y_train, epochs=5, validation_data=(x_test_pad, y_test), batch_size=32)

    # Save model and tokenizer
    model.save(model_file)
    with open(tokenizer_file, 'wb') as f:
        pickle.dump(tokenizer, f)
else:
    model = load_model(model_file)
    with open(tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)

# Step 3: Predict a message
def predict_message(message):
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=maxlen)
    prediction = model.predict(padded)[0][0]
    print("Message:", message)
    print("Prediction:", "SPAM" if prediction > 0.5 else "HAM")
    print(f"Confidence: {prediction:.2f}\n")

# Step 4: Try some example predictions
predict_message("Congratulations! You've won a free ticket to Bahamas. Call now!")
predict_message("Hey, are we still meeting at 6?")
predict_message("Win a brand new car by just sending YES to 12345!")
predict_message("Let's catch up over coffee tomorrow.")
