import streamlit as st
import warnings
import google.generativeai as genai
from PIL import Image

BOLD_BEGIN = "\033[1m"
BOLD_END = "\033[0m"

system_content = """If the image contains a face, detect the emotion of the person in the image through their facial expression using only one word and one emoji as "Angry 😠" or "Disgusted 🤢" or "Fearful 😨" or "Happy 😊" or "Neutral 😐" or "Sad 😢" or "Surprised 😲". Otherwise, Strictly return 'No Faces Detected' if no faces are present, regardless of objects that resemble faces."""

api_key = "api-key"  # Replace with your actual API key
model_name = "gemini-1.5-flash"

if not api_key:
    raise ValueError("API_KEY must be set.")

genai.configure(api_key=api_key)

class ClientFactory:
    def __init__(self):
        self.clients = {}
    
    def register_client(self, name, client_class):
        self.clients[name] = client_class
    
    def create_client(self, name, **kwargs):
        client_class = self.clients.get(name)
        if client_class:
            return client_class(**kwargs)
        raise ValueError(f"Client '{name}' is not registered.")

client_factory = ClientFactory()
client_factory.register_client('google', genai.GenerativeModel)

client_kwargs = {
    "model_name": model_name,
    "generation_config": {"temperature": 0.8},
    "system_instruction": None,
}

client = client_factory.create_client('google', **client_kwargs)

user_content = """If the image contains a face, detect the emotion of the person in the image through their facial expression using only one word and one emoji as "Angry 😠" or "Disgusted 🤢" or "Fearful 😨" or "Happy 😊" or "Neutral 😐" or "Sad 😢" or "Surprised 😲". Otherwise, Strictly return 'No Faces Detected' if no faces are present, regardless of objects that resemble faces."""

st.title("EmotiTrack")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")

    st.write("Detecting Expression...")
    response = client.generate_content([user_content, image], stream=True)
    response.resolve()
    st.write(f"{response.text}")
