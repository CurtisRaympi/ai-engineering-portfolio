import streamlit as st
import random
import re

st.set_page_config(page_title="NLP Chatbot", page_icon="🤖")

st.markdown(
    """
    <style>
    .bot {
        background-color: #e0f7fa;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 70%;
    }
    .user {
        background-color: #ffe0b2;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 70%;
        margin-left: auto;
    }
    .chat-container {
        max-width: 700px;
        margin: auto;
        background: #fafafa;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 0 10px #ccc;
    }
    .header {
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 10px;
        color: #0077b6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown('<div class="header">🤖 NLP Chatbot</div>', unsafe_allow_html=True)

# Define conversation patterns and responses
pairs = [
    (r"hi|hello|hey", ["Hello!", "Hi there!", "Hey! How can I assist you?"]),
    (r"how are you ?", ["I'm doing great, thanks!", "Feeling good! How about you?"]),
    (r"what is your name ?", ["I am your friendly chatbot.", "You can call me Chatbot."]),
    (r"what can you do ?", ["I can chat with you, answer simple questions, and help you learn NLP!"]),
    (r"tell me a joke", ["Why did the computer show up at work late? It had a hard drive!", "I would tell you a UDP joke, but you might not get it."]),
    (r"thank you|thanks", ["You're welcome!", "No problem!", "Glad to help!"]),
    (r"quit|exit|bye", ["Goodbye! Have a great day!", "See you later!", "Chat with you soon!"]),
    (r"(.*)", ["Sorry, I didn't understand that.", "Can you rephrase?", "I'm still learning. Can you try saying that differently?"])
]

def chatbot_response(user_input):
    for pattern, responses in pairs:
        if re.search(pattern, user_input, re.IGNORECASE):
            return random.choice(responses)
    return "I’m not sure how to respond to that."

# Chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", "")

if user_input:
    response = chatbot_response(user_input)
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("bot", response))

for sender, message in st.session_state.history:
    if sender == "user":
        st.markdown(f'<div class="user">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot">{message}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
