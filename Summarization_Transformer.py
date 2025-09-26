
# It will give the summary of the given text input. That is it will help to understand the context
# of the long paragraphs.

from transformers import pipeline

# Using the pretrained model for performing the summarization
# The bart model is trained on english language... so it will return the text in a grammatically correct
# format.
summary = pipeline("summarization","facebook/bart-large-cnn")

# defining the text input on which the summarization has to be performed

text = """ Nature’s beauty lies in its endless harmony, where every element seems to belong to a greater masterpiece. Towering mountains, draped in mist, stand proudly against skies painted with hues of sunrise and sunset. Rivers carve their way through valleys, reflecting the sparkle of sunlight like scattered diamonds. Forests whisper with the rustling of leaves, alive with the songs of birds and the gentle hum of life unseen. Meadows bloom with wildflowers, their colors blending into breathtaking tapestries. The sea, ever-changing, mirrors the moods of the sky above. In every corner, nature reminds us of peace, wonder, and timeless perfection. """

# Feeding the model the input
# specify the max_length/  min_length by default it will take max_length = 200, min_length= 30
result = summary(text)

print(result)

