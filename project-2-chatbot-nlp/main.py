# Chatbot with NLP - Using NLTK & TensorFlow
# Curtis Raympi

import nltk
from nltk.chat.util import Chat, reflections

# Sample pairs of patterns and responses
pairs = [
    [
        r"hi|hello|hey",
        ["Hello! How can I assist you today?", "Hi there! What can I do for you?"]
    ],
    [
        r"what is your name?",
        ["I am an AI chatbot created by Curtis."]
    ],
    [
        r"how are you?",
        ["I'm good, thanks for asking!", "Doing well, how about you?"]
    ],
    [
        r"quit",
        ["Goodbye! Have a great day."]
    ],
]

def chatbot():
    print("Chatbot is running! Type 'quit' to exit.")
    chat = Chat(pairs, reflections)
    chat.converse()

if __name__ == "__main__":
    nltk.download('punkt')
    chatbot()
