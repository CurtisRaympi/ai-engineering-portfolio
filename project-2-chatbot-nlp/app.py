import streamlit as st

st.set_page_config(page_title="🤖 AI Chatbot NLP", page_icon="🤖", layout="wide")

# Chatbot logic
def chatbot_response(user_msg):
    user_msg = user_msg.lower().strip()
    responses = {
        "hi": "Hello! 👋 How can I help you today?",
        "hello": "Hi there! What can I do for you?",
        "how are you": "I'm always running at 100%! How about you?",
        "help": "I can answer your questions about AI, projects, or just chat for fun!",
        "what can you do": "I can chat, tell jokes, answer AI questions, and more.",
        "tell me a joke": "Why did the AI go to therapy? Because it had too many neural issues! 🤖😂",
        "bye": "Goodbye! Have a wonderful day! 👋",
        "thank you": "You're welcome! Always here to help.",
    }
    return responses.get(user_msg, "Hmm, I’m not sure about that. Could you rephrase?")

# Keep chat history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("About This Chatbot")
    st.write("This chatbot is part of Curtis Raympi's AI Engineering Portfolio.")
    st.write("Built using Streamlit and basic NLP logic.")

# Main title
st.title("🤖 AI Chatbot with NLP")

# User input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Type something...")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state.chat_history.append(("user", user_input))
    bot_reply = chatbot_response(user_input)
    st.session_state.chat_history.append(("bot", bot_reply))

# Display chat
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.markdown(
            f"<div style='background:#DCF8C6;padding:10px;border-radius:10px;margin:5px 0'><b>You:</b> {message}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background:#E6E6E6;padding:10px;border-radius:10px;margin:5px 0'><b>Bot:</b> {message}</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown("Developed by **Emmanuel Jaja** | [GitHub](https://github.com/Emmanueljaja) | [LinkedIn](https://www.linkedin.com/in/Emmanuel jaja)")
