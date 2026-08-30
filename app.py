import base64
import os
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st

# Settings
DB_PATH = 'chroma_db'
COLLECTION_NAME = 'muneeb_portfolio'

# API Key Load
api_key = os.getenv("GOOGLE_API_KEY")

# 1. Gemini LLM Model Setup
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash", api_key=api_key, temperature=0
)

# 2. Page Settings
st.set_page_config(
    page_title='MUNEEB ULLAH | AI Portfolio',
    page_icon='assets/avatar.png',
    layout='centered',
)


# Background Image & Responsive Display Helper Function
def set_bg_image(image_path):
  if os.path.exists(image_path):
    with open(image_path, "rb") as image_file:
      encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        /* Main App Background */
        [data-testid="stAppViewContainer"] {{
            background-image: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)), url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: left center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Header Transparent */
        [data-testid="stHeader"] {{
            background-color: rgba(0, 0, 0, 0) !important;
        }}

        /* ===== DESKTOP DISPLAY (Screen width > 768px) ===== */
        @media (min-width: 769px) {{
            [data-testid="stMainBlockContainer"], .block-container {{
                max-width: 55% !important;
                margin-left: auto !important;
                margin-right: 4% !important;
                padding-top: 4rem !important;
            }}
        }}

        /* ===== MOBILE DISPLAY FIX (Screen width <= 768px) ===== */
        @media (max-width: 768px) {{
            [data-testid="stMainBlockContainer"], .block-container {{
                max-width: 95% !important;
                margin-left: auto !important;
                margin-right: auto !important;
                padding-top: 1rem !important;
                padding-left: 10px !important;
                padding-right: 10px !important;
            }}

            [data-testid="stAppViewContainer"] {{
                background-position: center top !important;
            }}

            h1, h3, p {{
                text-align: center !important;
                margin-left: 0px !important;
            }}

            h1 {{
                font-size: 1.8rem !important;
            }}

            h3 {{
                font-size: 1.2rem !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Function Call
set_bg_image("assets/bg.png")

# Custom CSS for Chat Alignment
st.markdown(
    """
<style>
/* 1. User Message (Right Side Align) */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]),
div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {
    flex-direction: row-reverse !important;
    text-align: right !important;
    margin-left: auto !important;
    max-width: 85% !important;
    background-color: #2b2d42 !important;
    border-radius: 18px 18px 2px 18px !important;
    padding: 10px !important;
}

/* 2. Assistant Message (Left Side Align) */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]),
div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) {
    margin-right: auto !important;
    max-width: 90% !important;
    background-color: #161b22 !important;
    border-radius: 18px 18px 18px 2px !important;
    padding: 10px !important;
}

/* Text right align fix for user input */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) p {
    text-align: right !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Title & Headers
st.markdown(
    """<h1 style="background: linear-gradient(45deg, white, gray);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;">MUNEEB ULLAH | AI Portfolio</h1>""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h3 style="
        background: linear-gradient(90deg, #38BDF8, #A855F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    ">
        Ask me anything about my portfolio!
    </h3>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="margin-left: 0px; color: #e0e0e0;">Ask me anything about my'
    ' skills, education, experience, projects and professional background.</p>',
    unsafe_allow_html=True,
)

# Check database
if not os.path.exists(DB_PATH):
  st.error("knowledge base not found!")
  st.info("Run: python create_database.py")
  st.stop()

# 3. Google AI Embeddings (Ollama Ki Jagah)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", google_api_key=api_key
)

# Load Chroma database
db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_PATH,
    embedding_function=embeddings,
)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 12})

# System prompt
SYSTEM_PROMPT = """
You are an expert portfolio assistant for Muneeb Ullah.
Your job is to answer ANY question about Muneeb using ONLY the provided Context.

Rules:
1. **Contact Information Handling:**
   - If user asks for phone number, email, address, or contact details in ANY language (English/Urdu/Roman Urdu), search all retrieved context thoroughly for "Contact Phone", "Primary Email", "+92...", "Khanapul", "Rawalpindi".
   - Phone (+92 318-0387132), Email (munibullah157@gmail.com), Location (Rawalpindi), LinkedIn, and GitHub are part of Muneeb's portfolio metadata. Always display them when asked.
2. **Language Matching:**
   - If user asks in Roman Urdu or Urdu, respond in Roman Urdu / Urdu.
   - If user asks in English, respond in English.
3. **Accuracy:**
   - Do not hallucinate. Summarize details clearly.
"""

# Chat History
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display old messages with custom avatars
for message in st.session_state.messages:
  avatar = "assets/avatar.png" if message["role"] == "assistant" else "👤"
  with st.chat_message(message["role"], avatar=avatar):
    st.markdown(message["content"])

# User input
question = st.chat_input("Ask about Muneeb...")

if question:
  # 1. Save & Display User Message
  st.session_state.messages.append({"role": "user", "content": question})
  with st.chat_message("user", avatar="👤"):
    st.markdown(question)

  # 2. Assistant Processing Block
  with st.chat_message("assistant", avatar="assets/avatar.png"):
    with st.spinner("Searching portfolio..."):
      try:
        # Retrieve documents
        documents = retriever.invoke(question)

        # Create context
        context = "\n\n".join([doc.page_content for doc in documents])

        # Create prompt
        prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}
Answer:"""

        # Get answer
        response = llm.invoke(prompt)

        # Clean text extract
        if isinstance(response.content, list):
          final_text = response.content[0].get("text", "")
        else:
          final_text = response.content

        # Show answer
        st.markdown(final_text)

        # Save answer in session history
        st.session_state.messages.append(
            {"role": "assistant", "content": final_text}
        )

      except Exception as e:
        st.error(f"Error: {e}")