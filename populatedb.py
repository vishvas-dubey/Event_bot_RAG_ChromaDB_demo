import streamlit as st
import os
import html
from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EventAssistantRAGBot:
    def __init__(self, api_key, chroma_path="chroma"):
        self.api_key = api_key
        self.chroma_path = chroma_path
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        self.prompt_template = """
        You are a friendly Event Information Assistant. Your primary purpose is to answer questions about the event described in the provided context. Follow these guidelines:

        1. You can respond to basic greetings like "hi", "hello", or "how are you" in a warm, welcoming manner
        2. For event information, only provide details that are present in the context
        3. If information is not in the context, politely say "I'm sorry, I don't have that specific information about the event"
        4. Keep responses concise but conversational
        5. Do not make assumptions beyond what's explicitly stated in the context
        6. Always prioritize factual accuracy while maintaining a helpful tone
        7. Do not introduce information that isn't in the context
        8. If unsure about any information, acknowledge uncertainty rather than guess
        9. You may suggest a few general questions users might want to ask about the event
        10. Remember to maintain a warm, friendly tone in all interactions
        11. You should refer to yourself as "Event Bot"

        Remember: While you can be conversational, your primary role is providing accurate information about this specific event based on the context provided.

        Context information about the event:
        {context}
        --------
        
        Now, please answer this question about the event: {question}
        """

    def post_process_response(self, response, query):
        """Format responses for better readability based on query type."""
        # If it's a lunch-related query, format the response
        if "lunch" in query.lower() or "food" in query.lower() or "eat" in query.lower():
            # Just start with "Regarding lunch:" without the greeting
            formatted = "Regarding lunch:\n\n"
            
            # Split into readable bullet points
            points = []
            
            # Extract key information using common phrases and format as separate points
            if "provided to all" in response:
                points.append("• Lunch will be provided to all participants who have checked in at the venue.")
            if "cafeteria" in response.lower() and "floor" in response.lower():
                # Extract time info if available
                time_info = ""
                if "1:00" in response and "2:00" in response:
                    time_info = "between 1:00 PM and 2:00 PM IST"
                points.append(f"• It will be served in the Cafeteria on the 5th floor {time_info}.")
            if "check-in" in response.lower() or "registration" in response.lower():
                points.append("• Please ensure you've completed the check-in process at the registration desk to be eligible.")
            if "volunteer" in response.lower() or "direction" in response.lower():
                points.append("• Feel free to ask a volunteer if you need directions to the cafeteria.")
                
            # If we couldn't extract structured points, just use the original
            if not points:
                return response
                
            # Combine all points with line breaks
            return formatted + "\n".join(points)
        
        # For other responses, just return the original
        return response

    def answer_question(self, query):
        """Use RAG with Google Gemini to answer a question based on retrieved context."""
        try:
            with st.spinner("Retrieving relevant information..."):
                # Using Google embeddings
                embedding_function = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=self.api_key
                )
                
                # Retrieve relevant documents from Chroma
                db = Chroma(persist_directory=self.chroma_path, embedding_function=embedding_function)
                results = db.similarity_search_with_score(query, k=5)
                context_text = "\n\n --- \n\n".join([doc.page_content for doc, _score in results])
                
                # Format the prompt
                prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
                prompt = prompt_template.format(context=context_text, question=query)
                
                # Create the content for Gemini
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                        ],
                    ),
                ]
            
            with st.spinner("Generating response..."):
                # Generate response using Gemini 2.0 Flash model
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",  # Using Gemini 2.0 Flash model
                    contents=contents,
                )
                
                raw_response = response.text
                return self.post_process_response(raw_response, query)
                
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Set page configuration
st.set_page_config(
    page_title="Build with AI - RAG Event Bot",
    page_icon="🎫",
    layout="centered"
)

# Load the CSS as before
st.markdown("""
<style>
/* Bot message formatting */
.bot-message {
    white-space: pre-line !important;
    line-height: 1.5 !important;
    margin-bottom: 0 !important;
}

.bot-message ol {
    margin-top: 8px !important;
    margin-bottom: 8px !important;
    padding-left: 25px !important;
}

.bot-message li {
    margin-bottom: 6px !important;
    padding-bottom: 0 !important;
    line-height: 1.4 !important;
}

.bot-message p {
    margin-bottom: 10px !important;
}

/* Chat container and message styles */
.custom-chat-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 20px;
    max-width: 800px;
}

.message-container {
    display: flex;
    align-items: flex-start;
    margin-bottom: 10px;
}

.message-container.user {
    flex-direction: row-reverse;
}

.avatar-icon {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background-color: #E8F0FE;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 20px;
    margin: 0 10px;
    flex-shrink: 0;
}

.user-avatar-icon {
    background-color: #F0F2F5;
}

.user-message {
    background-color: #F0F2F5;
    padding: 10px 15px;
    border-radius: 18px;
    max-width: 80%;
    margin-right: 10px;
    word-wrap: break-word;
}

.bot-message {
    background-color: #E8F0FE;
    padding: 10px 15px;
    border-radius: 18px;
    max-width: 80%;
    margin-left: 10px;
    word-wrap: break-word;
}
</style>
""", unsafe_allow_html=True)

# Main app title
st.title("Build with AI - RAG Event Bot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("API key not found in .env file. Please add GEMINI_API_KEY to your .env file.")
    st.stop()

# Initialize the bot
if "bot" not in st.session_state:
    chroma_path = "chroma"
    if not os.path.exists(chroma_path):
        st.error(f"Chroma directory '{chroma_path}' not found. Make sure your vector database is properly set up.")
        st.stop()
    
    with st.spinner("Initializing assistant..."):
        st.session_state.bot = EventAssistantRAGBot(api_key, chroma_path)
    
    # Add welcome message with options
    if not st.session_state.messages:
        welcome_message = """Hello! I'm Event bot.
I can help you with the following:
1. Agenda of the "Build with AI" workshop
2. Important Dates of this workshop
3. Details of the AI Hackathon
4. Presentation of Interesting projects in AI, ML
5. Locating the washrooms
6. Details of lunch at the venue

How can I help you with information about this event?"""
        
        st.session_state.messages.append(
            {"role": "assistant", "content": welcome_message}
        )

# Custom Chat UI Implementation - Same as previous example
# Open a container div for the chat
chat_html = '<div class="custom-chat-container">'

# Add all messages to the custom chat HTML
for message in st.session_state.messages:
    if message["role"] == "user":
        avatar = '<div class="avatar-icon user-avatar-icon">👤</div>'
        chat_html += f'<div class="message-container user">'
        chat_html += avatar
        chat_html += f'<div class="user-message">{html.escape(message["content"])}</div>'
        chat_html += '</div>'
    else:  # assistant
        avatar = '<div class="avatar-icon">🤖</div>'
        chat_html += f'<div class="message-container">'
        chat_html += avatar
        # Format the welcome message to be more compact
        formatted_content = message["content"]
        if "I can help you with the following:" in formatted_content:
            # Replace the original formatting with HTML formatting
            formatted_content = formatted_content.replace("Hello! I'm Event bot.\nI can help you with the following:", 
                                                          "Hello! I'm Event bot.<br><br>I can help you with the following:")
            formatted_content = formatted_content.replace("\n1. ", "<ol style='margin-top:8px;margin-bottom:8px;padding-left:25px;'><li style='margin-bottom:4px;'>")
            formatted_content = formatted_content.replace("\n2. ", "</li><li style='margin-bottom:4px;'>")
            formatted_content = formatted_content.replace("\n3. ", "</li><li style='margin-bottom:4px;'>")
            formatted_content = formatted_content.replace("\n4. ", "</li><li style='margin-bottom:4px;'>")
            formatted_content = formatted_content.replace("\n5. ", "</li><li style='margin-bottom:4px;'>")
            formatted_content = formatted_content.replace("\n6. ", "</li><li style='margin-bottom:4px;'>")
            formatted_content = formatted_content.replace("\n\nHow can I help you", "</li></ol><br>How can I help you")
            chat_html += f'<div class="bot-message">{formatted_content}</div>'
        else:
            chat_html += f'<div class="bot-message">{html.escape(message["content"]).replace("\n", "<br>")}</div>'
        chat_html += '</div>'

# Close the container div
chat_html += '</div>'

# Render the custom chat container
st.markdown(chat_html, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask a question about the event...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response
    response = st.session_state.bot.answer_question(user_input)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update the UI
    st.rerun()