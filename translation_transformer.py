from transformers import pipeline

# For French To English
# translator = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")

# For English To French
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

text = "How are you?"
# text = "Comment allez-vous ?"

translation = translator(text)

#print("English:", translation[0]['translation_text'])
print("French:", translation[0]['translation_text'])