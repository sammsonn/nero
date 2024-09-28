import streamlit as st
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
import warnings
import tempfile
import os

# Filter out warnings
warnings.filterwarnings("ignore")

# Set OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

st.title(":fire: Ask Nero:")
st.divider()


def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url("https://getwallpapers.com/wallpaper/full/5/2/e/351848.jpg");
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


set_bg_hack_url()


@st.cache_resource(show_spinner=False)
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    raw_doc = loader.load()
    os.unlink(tmp_path)  # Delete the temp file after loading

    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    documents = splitter.split_documents(raw_doc)
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    vectorstore = Chroma.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file is not None:
    retriever = process_pdf(uploaded_file)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Accept user input
    user_input = st.chat_input("Ask away...")
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Retrieve relevant document content if a document has been uploaded
        sources = retriever.get_relevant_documents(user_input)
        context = ""
        for doc in sources:
            context += doc.page_content + "\n\n"
        augmented_user_input = "Context: \""" " + context + "\"""\n\nQuestion: " + user_input + "\n"

        # Chat with GPT using augmented input
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Use the provided articles"
                                              " that are delimited by triple quotes to answer questions and"
                                              " always specify the relevant source phrase from the article."
                                              " If the answer cannot be found in the articles,"
                                              " write 'I could not find an answer.'."},
                {"role": "user", "content": augmented_user_input}
            ]
        )

        response_text = response.choices[0].message.content.strip()

        # Add assistant message to chat history and display
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant", avatar="🔥"):
            st.markdown(response_text)
else:
    st.write("Please upload a PDF document to proceed.")
