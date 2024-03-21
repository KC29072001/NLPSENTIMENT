
import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset 


# Load tokenizer and model
model = TFAutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load emotion dataset
emotions = load_dataset('SetFit/emotion')

# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True)



emotions_encoded.set_format('tensorflow', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

# Set batch size
BATCH_SIZE = 64

def order(input_ids, attention_mask, token_type_ids, label):
    '''
    This function will group all the inputs of BERT
    into a single dictionary and then output it with
    labels.
    '''
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }, label


# Converting train split of `emotions_encoded` to tensorflow format
train_dataset = tf.data.Dataset.from_tensor_slices((emotions_encoded['train']['input_ids'],
                                                    emotions_encoded['train']['attention_mask'],
                                                    emotions_encoded['train']['token_type_ids'],
                                                    emotions_encoded['train']['label']))
# Set batch_size and shuffle
# train_dataset = train_dataset.map(order).batch(BATCH_SIZE).shuffle(1000)
train_dataset = train_dataset.map(order)



# Define BERT-based classification model
class BERTForClassification(tf.keras.Model):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        x = self.bert(inputs)[1]
        return self.fc(x)

# Instantiate the model
classifier = BERTForClassification(model, num_classes=6)

# Compile the model
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Streamlit app
def main():
    st.title("Emotion Detection with Transformers")

    sentence = st.text_input("Enter a sentence:")
    # emotion_labels = ["optimism", "surprise", "joy", "trust", "sadness", "anger"]
    if st.button("Predict"):
        if sentence:
            input_ids = tokenizer.encode(sentence, return_tensors="tf", max_length=128, truncation=True)
            # classifier.evaluate(test_dataset)
            predictions = classifier.predict(input_ids)
            # predicted_emotion_index = tf.argmax(predictions, axis=1).numpy()[0]
            # predicted_emotion = emotions['train']['features']['label_text'][predicted_emotion_index]
            # predicted_emotion = emotions['train'].features['label_text'][predicted_emotion_index]
            # st.write(f"Predicted Emotion: {predicted_emotion}")
            st.write(f"Predicted Probabilities: {predictions}")
            

if __name__ == "__main__":
    main()


