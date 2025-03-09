import os
import base64
import streamlit as st
from PIL import Image
from groq import Groq
from dotenv import load_dotenv


# Load environment variables from .env
# load_dotenv()

# # Get API key from .env file
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY is not set. Check your .env file.")

# client = Groq(api_key=GROQ_API_KEY)


# Get API key from Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing! Set it in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# st.write("App is running successfully!")



# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to generate caption for the uploaded image using Groq
def generate_caption(image_path):
    # Get the base64-encoded string of the image
    base64_image = encode_image(image_path)
    
    # Send image to Groq for captioning
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},  # User request
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",  # Image in base64 format
                    }
                },
            ]
        }],
        model="llama-3.2-11b-vision-preview",  # Use the vision model
    )
    
    # Correct way to access the response (accessing the 'choices' attribute)
    # Assuming Groq's ChatCompletion response has a 'choices' attribute directly accessible
    response_content = response.choices[0].message.content
    return response_content

# Function to handle chat based on the image caption
def chat_with_groq(chat_history, user_input, image_caption):
    # System prompt using the image caption
    system_prompt = f"""
    You are an AI assistant that answers questions based on an image.
    The image is described as: '{image_caption}'. 
    Only answer questions related to the image. If a question is not relevant to the image, politely decline.
    """
    
    # Construct message history
    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": "user", "content": msg} for msg in chat_history]
    messages.append({"role": "user", "content": user_input})
    
    # Get response from Groq
    response = client.chat.completions.create(
        messages=messages,
        model="llama-3.2-11b-vision-preview",  # Use the same vision model
    )
    
    # Correct way to access the response (accessing the 'choices' attribute)
    return response.choices[0].message.content

# Main Streamlit app
def main():
    st.title("Image Captioning and Chat Assistant")
    
    # File uploader for the image
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Save the image locally
        image_path = f"temp_{uploaded_image.name}"
        image.save(image_path)
        
        # Generate caption for the image
        st.write("Generating caption for the image...")
        image_caption = generate_caption(image_path)
        st.write(f"**Caption:** {image_caption}")
        
        # Chat interaction
        st.write("---")
        st.subheader("Chat About the Image")
        chat_history = []
        
        # Input box for user query
        user_input = st.text_input("Ask something about the image:")
        if user_input:
            response = chat_with_groq(chat_history, user_input, image_caption)
            chat_history.append(user_input)  # Add user input to chat history
            chat_history.append(response)  # Add AI response to chat history
            
            # Display chat history
            for i, message in enumerate(chat_history):
                if i % 2 == 0:  # User message
                    st.write(f"**You:** {message}")
                else:  # AI response
                    st.write(f"**AI:** {message}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
