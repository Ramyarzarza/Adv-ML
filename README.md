
# Phishing Email Detection: Explainability and Model Sensitivity with LIME

## Introduction to LIME (Local Interpretable Model-Agnostic Explanations)

LIME, or **Local Interpretable Model-Agnostic Explanations**, is an XAI (Explainable AI) technique designed to make machine learning models more interpretable. It works by generating locally faithful explanations for individual predictions of any black-box model. This means that LIME tries to understand the model’s decision for one specific instance, such as why an email is classified as phishing.

LIME is model-agnostic, meaning it can be applied to any machine learning model, whether it's a simple logistic regression or a complex neural network. By perturbing the input data (i.e., making small changes to the email text) and analyzing the impact on the prediction, LIME provides insights into which parts of the input were most important for the model's decision.

In this project, we use LIME to explain the predictions of three models:
- Random Forest with tokenized email text.
- Random Forest with embedding.
- LSTM for sequential analysis of text.

---

## Phishing Email Detection: Explainability and Model Sensitivity

This project aims to detect phishing emails using different models: Random Forest with tokenization, Random Forest with embedding, and LSTM. We also investigate how Explainable AI (XAI) techniques, such as LIME, can help us understand and interpret the predictions of these models. In particular, we analyze how the models react when a clearly phishing email is modified with a few positive words added at the end.

## 1. Random Forest with Tokenizing

### Model Training
The email text is tokenized and converted into sequences of integers, which are then used as features for a Random Forest model.

```python
# Tokenizing the emails
sequences = tokenizer.texts_to_sequences(emails)
padded_sequences = pad_sequences(sequences, padding='post', maxlen=100)

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(padded_sequences, y_train)
```

### Sensitivity Analysis
We used a phishing email with a high confidence prediction and added a few positive words to see how sensitive the Random Forest model is to this change.

```python
# Prediction before adding positive words
print(rf_model.predict([padded_sequences[4]]))

# Modifying the email by adding positive words
modified_email = padded_sequences[4] + " thank you for your help"
print(rf_model.predict([modified_email]))
```

### LIME Explanation
LIME was used to explain the predictions of the Random Forest model by highlighting the words that contributed most to the phishing classification.

```python
# LIME Explanation
explainer = lime_text.LimeTextExplainer(class_names=['Safe Email', 'Phishing Email'])
exp = explainer.explain_instance(text_instance, rf_model.predict_proba)
exp.show_in_notebook()
```

## 2. Random Forest using Embedding

### Model Training
In this approach, we use an embedding layer to convert the tokenized text into dense vectors before applying the Random Forest model.

```python
# Embedding the sequences
embedding_dim = 100
embedding_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=embedding_dim, input_length=100)
X_embedded = embedding_layer(padded_sequences)

# Random Forest model with embedding
rf_model_emb = RandomForestClassifier(n_estimators=100)
rf_model_emb.fit(X_embedded, y_train)
```

### Sensitivity Analysis
We added positive words to the phishing email to analyze how the model's prediction changes when using embeddings.

```python
# Prediction before modification
print(rf_model_emb.predict([X_embedded[4]]))

# Modifying the email
modified_email_embedded = X_embedded[4] + " thank you for your support"
print(rf_model_emb.predict([modified_email_embedded]))
```

### LIME Explanation
LIME was used again to interpret the predictions of the Random Forest model with embedding, showing which parts of the email influenced the decision.

```python
# LIME Explanation for embedding
exp_emb = explainer.explain_instance(text_instance, rf_model_emb.predict_proba)
exp_emb.show_in_notebook()
```

## 3. LSTM

### Model Training
In this approach, we use an LSTM (Long Short-Term Memory) model, which is well-suited for sequential data like text.

```python
# LSTM model definition and training
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(100, embedding_dim)),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
```

### Sensitivity Analysis
We used the LSTM model to predict the phishing email both before and after adding positive words.

```python
# Original email prediction
original_prediction = model.predict([X_test[4]])
print(f"Original prediction: {original_prediction}")

# Modified email prediction
modified_prediction = model.predict([modified_email])
print(f"Modified prediction: {modified_prediction}")
```

### LIME Explanation
Using LIME, we explain the LSTM model's predictions, demonstrating which words and patterns the model considered most important.

```python
# LIME Explanation for LSTM
exp_lstm = explainer.explain_instance(text_instance, model.predict)
exp_lstm.show_in_notebook()
```

## 4. Comparison and XAI Insights

### Model Sensitivity
- **Random Forest with Tokenizing**: This model tends to be more sensitive to specific words, and adding positive words to a phishing email can sometimes alter the prediction.
- **Random Forest with Embedding**: The embedding layer makes the model less sensitive to individual words and more robust to changes.
- **LSTM**: The LSTM model is more resilient to minor text changes due to its ability to capture long-term dependencies in the text.

### Explainability with LIME
LIME provided explanations for each model:
- **Random Forest with Tokenizing**: LIME showed which words the model relied on to predict phishing.
- **Random Forest with Embedding**: LIME revealed how the embedding layer allowed the model to generalize better.
- **LSTM**: LIME helped explain the broader patterns captured by the LSTM model.

## Conclusion
Explainable AI (XAI) techniques like LIME provide a clear way to interpret the decision-making process of black-box models. By leveraging XAI, we can understand how sensitive models are to changes in input text, and we can better explain the reasoning behind predictions.

