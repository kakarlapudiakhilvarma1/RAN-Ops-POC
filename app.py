import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict
import re
from datetime import datetime
import uuid

# Langchain and RAG imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage

# Load environment variables
load_dotenv()

# Configuration variables
image = os.getenv('LOGO_PATH')
#LANGCHAIN_TRACING_V2 = "true"
#LANGCHAIN_API_KEY = os.getenv('LANGSMITH_API_KEY')
#LANGCHAIN_PROJECT = "noc-poc-rag-tracing"

# Language configuration
SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "English": {
        "code": "en",
        "welcome": """
        👋 Welcome to RAN Ops Assist! 
        
        I'm your AI-powered NOC (Network Operations Center) assistant, specialized in Radio Access Network (RAN) operations. 
        
        I can help you with:
        - Troubleshooting network issues
        - Providing insights on alarms and incidents
        - Guiding you through NOC best practices
        
        How can I assist you today with your telecom network operations?
        """
    },
    "Romanian": {
        "code": "ro",
        "welcome": """
        👋 Bun venit la RAN Ops Assist! 
        
        Sunt asistentul dvs. NOC (Network Operations Center) bazat pe AI, specializat în operațiuni Radio Access Network (RAN). 
        
        Vă pot ajuta cu:
        - Depanarea problemelor de rețea
        - Oferirea de informații despre alarme și incidente
        - Ghidarea prin cele mai bune practici NOC
        
        Cum vă pot ajuta astăzi cu operațiunile dvs. de rețea de telecomunicații?
        """
    },
    "German": {
        "code": "de",
        "welcome": """
        👋 Willkommen bei RAN Ops Assist! 
        
        Ich bin Ihr KI-gestützter NOC (Network Operations Center) Assistent, spezialisiert auf Radio Access Network (RAN) Betrieb. 
        
        Ich kann Ihnen helfen bei:
        - Fehlerbehebung von Netzwerkproblemen
        - Einblicke in Alarme und Vorfälle
        - Anleitung durch NOC Best Practices
        
        Wie kann ich Ihnen heute bei Ihren Telekommunikationsnetzwerk-Operationen helfen?
        """
    }
}

def generate_chat_id():
    """Generate a unique chat ID."""
    return str(uuid.uuid4())[:8]

def get_timestamp():
    """Get current timestamp in readable format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def initialize_session_state(language: str):
    """Initialize session state with welcome message and chat management."""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    
    if "current_chat_id" not in st.session_state:
        new_chat_id = generate_chat_id()
        st.session_state.current_chat_id = new_chat_id
        st.session_state.chats[new_chat_id] = {
            "messages": [{
                "role": "assistant",
                "content": SUPPORTED_LANGUAGES[language]["welcome"]
            }],
            "timestamp": get_timestamp(),
            "title": "New Chat"
        }
    
    if "selected_language" not in st.session_state:
        st.session_state.selected_language = language

def update_chat_title(chat_id: str, messages: List[dict]):
    """Update chat title based on the first user message."""
    for msg in messages:
        if msg["role"] == "user":
            title = msg["content"][:30] + "..." if len(msg["content"]) > 30 else msg["content"]
            st.session_state.chats[chat_id]["title"] = title
            break

def render_sidebar():
    """Render the sidebar with configuration and chat history."""
    with st.sidebar:
        st.header("Config")
        
        # Language selector
        selected_language = st.selectbox(
            "Select Language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.selected_language)
        )
        
        # Update selected language if changed
        if selected_language != st.session_state.selected_language:
            st.session_state.selected_language = selected_language
            st.rerun()
        
        st.header("Chat History")
        
        # New Chat button
        if st.button("New Chat", key="new_chat"):
            new_chat_id = generate_chat_id()
            st.session_state.chats[new_chat_id] = {
                "messages": [{
                    "role": "assistant",
                    "content": SUPPORTED_LANGUAGES[selected_language]["welcome"]
                }],
                "timestamp": get_timestamp(),
                "title": "New Chat"
            }
            st.session_state.current_chat_id = new_chat_id
            st.rerun()
        
        # Display chat history
        st.divider()
        for chat_id, chat_data in sorted(
            st.session_state.chats.items(),
            key=lambda x: x[1]["timestamp"],
            reverse=True
        ):
            chat_title = chat_data["title"]
            if st.button(
                f"{chat_title}\n{chat_data['timestamp']}",
                key=f"chat_{chat_id}",
                use_container_width=True
            ):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        
        return selected_language

def is_alarm_related_question(question: str) -> bool:
    """Check if the question is related to alarms or technical issues."""
    alarm_keywords = [
        'alarm', 'alert', 'error', 'failure', 'maintenance', 'connection',
        'unit', 'rf', 'radio', 'network', 'fault', 'down', 'offline', 'missing'
    ]
    return any(keyword in question.lower() for keyword in alarm_keywords)

def is_history_related_question(question: str) -> bool:
    """Check if the question is about chat history."""
    history_keywords = [
        'previous', 'earlier', 'before', 'last time', 'history',
        'what did', 'what was', 'what were', 'asked', 'said'
    ]
    return any(keyword in question.lower() for keyword in history_keywords)

@st.cache_resource
def setup_rag_components():
    """Initialize and cache RAG components."""
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    path = "pdf files"
    loader = PyPDFDirectoryLoader(path)
    extracted_docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splits.split_documents(extracted_docs)
    vector_store = FAISS.from_documents(documents=docs, embedding=embedding)
    return vector_store.as_retriever()

def format_chat_history(messages: List[dict]) -> str:
    """Format chat history into a string for the prompt."""
    formatted_history = []
    for msg in messages[1:]:  # Skip the initial greeting
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted_history.append(f"{role}: {msg['content']}")
    return "\n".join(formatted_history)

def create_rag_chain(llm, retriever, language: str):
    """Create RAG chains with different prompts for different types of questions."""
    # Alarm-related prompt
    alarm_prompt = ChatPromptTemplate.from_template(
        f"""
        You are a Telecom NOC Engineer with expertise in Radio Access Networks (RAN).
        Always respond in only {language} also always response should be in the structed format as mentioned.
        
        Previous conversation history:
        {{chat_history}}
        
        Current context: {{context}}
        Current question: {{input}}
        
        Response should be in short format and follow this structured format:
            1. Response: Provide an answer based on the given situation, with slight improvements for clarity but from the context.
            2. Explanation of the issue: Include a brief explanation on why the issue might have occurred.
            3. Recommended steps/actions: Suggest further steps to resolve the issue.
            4. Quality steps to follow:
                - Check for relevant INC/CRQ tickets.
                - Follow the TSDANC format while creating INC.
                - Mention previous closed INC/CRQ information if applicable.
                - If there are >= 4 INCs on the same issue within 90 days, highlight the ticket to the SAM-SICC team and provide all relevant details.
        """
    )
    
    # General conversation prompt
    general_prompt = ChatPromptTemplate.from_template(
        f"""
        You are a helpful NOC assistant.
        Always respond in {language}.
        
        Previous conversation history:
        {{chat_history}}
        
        Current context: {{context}}
        Current question: {{input}}
        
        Provide a natural, conversational response without following any specific format. 
        If the question is about chat history, give a brief and direct answer about previous interactions.
        Keep the response concise and relevant to the question asked.
        Please respond only if the question is related to history, context, telecom related, from knowledge base
        questions only. Don't answer questions which are not related to NOC Telecom operations.
        """
    )
    
    # Create chains
    alarm_chain = create_stuff_documents_chain(llm, alarm_prompt)
    general_chain = create_stuff_documents_chain(llm, general_prompt)
    
    return {
        'alarm': create_retrieval_chain(retriever, alarm_chain),
        'general': create_retrieval_chain(retriever, general_chain)
    }

def main():
    """Main application logic."""
    # Check for API key
    #google_api_key = os.getenv('GEMINI_API_KEY')
    google_api_key = st.text_input("Enter Gemini API KEY", type="password")
    if not google_api_key:
        st.info("Please add your Google AI API key to continue.", icon="🗝️")
        return

    # Streamlit page configuration
    #st.set_page_config(page_title="NOC Assist RAG Chatbot", page_icon="🔍")
    st.title("RAN Ops Assist 🔍📡")
    st.info('Always follow Quality Points', icon="ℹ️")

    # Initialize session state with default language
    initialize_session_state("English")
    
    try:
        # Configure Gemini
        genai.configure(api_key=google_api_key)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            convert_system_message_to_human=True
        )
        
        # Setup RAG components
        retriever = setup_rag_components()
        
        # Render sidebar and get selected language
        selected_language = render_sidebar()
        
        # Get current chat messages
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        messages = current_chat["messages"]
        
        # Create chains with selected language
        chains = create_rag_chain(llm, retriever, selected_language)
        
        # Display chat history
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input
        if prompt := st.chat_input("What would you like to know about NOC operations?"):
            # Add user message to chat
            messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                # Update chat title if this is the first user message
                update_chat_title(st.session_state.current_chat_id, messages)
                
                # Format chat history
                chat_history = format_chat_history(messages)
                
                # Choose appropriate chain based on question type
                if is_alarm_related_question(prompt):
                    response = chains['alarm'].invoke({
                        "input": prompt,
                        "chat_history": chat_history
                    })
                else:
                    response = chains['general'].invoke({
                        "input": prompt,
                        "chat_history": chat_history
                    })
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response['answer'])
                
                # Store assistant response
                messages.append({
                    "role": "assistant", 
                    "content": response['answer']
                })

            except Exception as e:
                st.error(f"An error occurred while generating response: {e}")
                
    except Exception as e:
        st.error(f"Error setting up RAG components: {e}")

if __name__ == "__main__":
    main()
