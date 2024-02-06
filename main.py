from transformers import TFAutoModelWithLMHead, AutoTokenizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load Turkish GPT-2 Model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/gpt2-turkish")
model = TFAutoModelWithLMHead.from_pretrained("dbmdz/gpt2-turkish")

# Function for sentence generation
def generate_sentence(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='tf')
    outputs = model.generate(inputs, max_length=50, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0])

# Example usage
triad = "çocuk inek hava"
print(generate_sentence(triad))



# Preprocessed Turkish corpus data
# X_train, Y_train represent your tokenized and sequence-labeled training data
X_train, Y_train = [], []

# LSTM Model Creation
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128)) # Assume vocabulary size is 10000
model.add(LSTM(128))
model.add(Dense(10000, activation='softmax')) # Assume vocabulary size is 10000
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Model Training
model.fit(X_train, Y_train, epochs=5)

# Sentence generation would require a seed sequence and generating words in a loop
def generate_sentence_lstm(model, seed_triad):
    output_sentence = seed_triad
    for _ in range(50): # Generate 50 words
        token_list = tokenizer.texts_to_sequences([output_sentence])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        output_sentence += " " + output_word
    return output_sentence

# Example usage
triad = "çocuk inek hava"
print(generate_sentence_lstm(model, triad))

