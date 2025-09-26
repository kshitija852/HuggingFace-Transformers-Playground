# Performing the spam classification.... for classifying if it is a spam or not
# Importing the Hugging_Face transformer library

from transformers import pipeline

# Loading the model which is trained with the spam data
spam_classifier = pipeline("text-classification", model= "philschmid/distilbert-base-multilingual-cased-sentiment")

texts = [
          "Congratulations!! You have won 500 usd amazon gift card.",
          "Urgent!! Your gmail account has been compromised.",
          "Hi, Amit let us have meeting tomorrow at 12 pm."
]

results = spam_classifier(texts)

label_mapping = {'negative':'SPAM','neutral':'NOT SPAM','positive':'NOT SPAM'}

for result in results:
    label = label_mapping[result['label']]
    score = result['score']
    print(f"Label:{label},Confidence: {score:.4f}")





