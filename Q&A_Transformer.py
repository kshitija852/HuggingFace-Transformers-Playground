# Implementing the Question-Answer using transformers

from transformers import pipeline

# The squad models are the models which are pretrained on question-answer
# context paragraph + a question, the model learns to point to the exact span of text inside the context.
question_answer = pipeline("question-answering",model="bert-large-uncased-whole-word-masking-finetuned-squad")

# As the squad models are pretrained on context paragraph + a question pattern we need to provide a context
# to get the answer
# If you want direct answers without providing a context, you need a generative model
# (e.g., GPT-style LLMs or T5-based models)
context = """Meditation offers a wide range of benefits for both the mind and body. By practicing regularly, it helps reduce stress and anxiety, creating a sense of inner calm and balance. It improves focus and concentration, allowing you to stay present and more productive in daily tasks. Meditation also promotes emotional well-being by fostering self-awareness and resilience, making it easier to manage negative thoughts and emotions. Physically, it can lower blood pressure, enhance sleep quality, and support overall health. Over time, meditation encourages a deeper connection between mind and body, leading to greater peace, clarity, and overall life satisfaction. """

result = question_answer(question="What is the main benefit of practicing meditation?" ,context=context)

# The model goes via the context and then it returns the result which matches best with the context.
print(f'result: {result}')


