import streamlit as st
import re

# Basic pattern-response pairs
pairs = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!', 'Hey!']),
    (r'how are you ?', ['I am doing well, thank you.', 'I am fine, how about you?']),
    (r'what is your name ?', ['I am Curtis\'s Chatbot.', 'You can call me Chatbot.']),
    (r'quit', ['Bye!', 'See you later!']),
]

def chatbot_response(user_input):
    for pattern, responses in pairs:
        if re.search(pattern, user_input.lower()):
            return responses[0]  # Just return the first matching response
    return "Sorry, I don't understand."

st.title("💬 Simple Chatbot with NLP")

user_input = st.text_input("You:", "")

if user_input:
    response = chatbot_response(user_input)
    st.text_area("Bot:", value=response, height=100)
