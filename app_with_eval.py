import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict, Tuple, Any
import re
from datetime import datetime
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import json
import time

# Langchain and RAG imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, Document

# RAGAS evaluation imports
from langchain_core.outputs import Generation
import numpy as np

# Load environment variables
load_dotenv()

# Configuration variables
image = os.getenv('LOGO_PATH')

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

# RAGAS Evaluation Implementation with Gemini
class GeminiRagasEvaluator:
    def __init__(self, google_api_key: str):
        genai.configure(api_key=google_api_key)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            convert_system_message_to_human=True
        )
    
    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Evaluate if the answer is faithful to the given contexts
        Returns a score between 0 and 1
        """
        prompt = f"""
        You are a critical evaluator assessing the faithfulness of an answer to provided context.

        Context:
        {' '.join(contexts)}

        Answer:
        {answer}

        Task:
        On a scale of 0 to 1, where 1 means the answer is completely faithful to the context and 0 means it contains hallucinations or unsupported information:
        1. Analyze each claim or statement in the answer
        2. Check if it's directly supported by the context
        3. Determine if there are any unsupported extrapolations
        4. Provide a single numerical score between 0 and 1

        Return only the numerical score without any explanation.
        """
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        score_text = response.text.strip()
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
        except ValueError:
            st.error(f"Failed to parse faithfulness score: {score_text}")
            return 0.5  # Default middle score

    def evaluate_relevance(self, question: str, contexts: List[str]) -> float:
        """
        Evaluate if the retrieved contexts are relevant to the question
        Returns a score between 0 and 1
        """
        prompt = f"""
        You are evaluating the relevance of retrieved documents to a question.

        Question:
        {question}

        Retrieved documents:
        {' '.join(contexts)}

        Task:
        On a scale of 0 to 1, where 1 means the documents are highly relevant to answering the question and 0 means they are completely irrelevant:
        1. Assess how well the documents address the information needs in the question
        2. Consider whether key information required to answer the question is present
        3. Ignore extraneous information if the core relevant content is present
        4. Provide a single numerical score between 0 and 1

        Return only the numerical score without any explanation.
        """
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        score_text = response.text.strip()
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
        except ValueError:
            st.error(f"Failed to parse relevance score: {score_text}")
            return 0.5  # Default middle score

    def evaluate_contextual_precision(self, answer: str, question: str, contexts: List[str]) -> float:
        """
        Evaluate if the answer uses relevant parts of the contexts efficiently
        Returns a score between 0 and 1
        """
        prompt = f"""
        You are evaluating the contextual precision of an answer.

        Question:
        {question}

        Answer:
        {answer}

        Contexts:
        {' '.join(contexts)}

        Task:
        On a scale of 0 to 1, where 1 means the answer efficiently uses only relevant parts of the context and 0 means it includes lots of irrelevant information:
        1. Determine how much of the context used in the answer was directly relevant to the question
        2. Check if the answer contains information from the context that doesn't help answer the question
        3. Assess whether the answer is concise while covering the necessary information
        4. Provide a single numerical score between 0 and 1

        Return only the numerical score without any explanation.
        """
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        score_text = response.text.strip()
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
        except ValueError:
            st.error(f"Failed to parse contextual precision score: {score_text}")
            return 0.5  # Default middle score

    def evaluate_answer_correctness(self, answer: str, ground_truth: str) -> float:
        """
        Evaluate if the answer is correct compared to the ground truth
        Returns a score between 0 and 1
        """
        prompt = f"""
        You are evaluating the correctness of an answer against a known ground truth.

        Answer to evaluate:
        {answer}

        Ground truth:
        {ground_truth}

        Task:
        On a scale of 0 to 1, where 1 means the answer completely matches the ground truth in meaning and information and 0 means it's completely incorrect:
        1. Compare the key information points in both texts
        2. Check for any contradictions or incorrect information
        3. Consider semantic equivalence rather than exact wording
        4. Provide a single numerical score between 0 and 1

        Return only the numerical score without any explanation.
        """
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        score_text = response.text.strip()
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
        except ValueError:
            st.error(f"Failed to parse correctness score: {score_text}")
            return 0.5  # Default middle score

    def evaluate_rag(self, question: str, answer: str, contexts: List[str], ground_truth: str = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of a RAG system response
        """
        results = {
            "faithfulness": self.evaluate_faithfulness(answer, contexts),
            "relevance": self.evaluate_relevance(question, contexts),
            "contextual_precision": self.evaluate_contextual_precision(answer, question, contexts),
        }
        
        # Only evaluate correctness if ground truth is provided
        if ground_truth:
            results["answer_correctness"] = self.evaluate_answer_correctness(answer, ground_truth)
        
        # Calculate average score
        results["average_score"] = sum(results.values()) / len(results)
        
        return results

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
    
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = []
    
    if "evaluation_mode" not in st.session_state:
        st.session_state.evaluation_mode = False
        
    if "awaiting_evaluation" not in st.session_state:
        st.session_state.awaiting_evaluation = False
        
    if "current_evaluation_data" not in st.session_state:
        st.session_state.current_evaluation_data = None
        
    if "evaluation_complete" not in st.session_state:
        st.session_state.evaluation_complete = False
        
    if "evaluation_results_data" not in st.session_state:
        st.session_state.evaluation_results_data = None

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

        # Evaluation mode toggle
        eval_mode = st.toggle("Evaluation Mode", value=st.session_state.evaluation_mode)
        if eval_mode != st.session_state.evaluation_mode:
            st.session_state.evaluation_mode = eval_mode
            # Reset evaluation states when toggling
            st.session_state.awaiting_evaluation = False
            st.session_state.current_evaluation_data = None
            st.session_state.evaluation_complete = False
            st.session_state.evaluation_results_data = None
            st.rerun()
        
        if st.session_state.evaluation_mode:
            st.info("In evaluation mode, you'll be asked to provide ground truth answers for evaluation")
        
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
            # Reset evaluation states for new chat
            st.session_state.awaiting_evaluation = False
            st.session_state.current_evaluation_data = None
            st.session_state.evaluation_complete = False
            st.session_state.evaluation_results_data = None
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
                # Reset evaluation states when switching chats
                st.session_state.awaiting_evaluation = False
                st.session_state.current_evaluation_data = None
                st.session_state.evaluation_complete = False
                st.session_state.evaluation_results_data = None
                st.rerun()
        
        if len(st.session_state.evaluation_results) > 0:
            st.header("Evaluation Dashboard")
            if st.button("View Evaluation Results", key="view_eval"):
                st.session_state.view_evaluation = True
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
    return vector_store.as_retriever(), docs

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

def render_evaluation_dashboard():
    """Render the evaluation dashboard with visualization of results."""
    st.title("RAG System Evaluation Dashboard")
    
    if not st.session_state.evaluation_results:
        st.warning("No evaluation results available. Run some evaluations first!")
        return
    
    # Convert results to DataFrame for easier analysis
    eval_df = pd.DataFrame(st.session_state.evaluation_results)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg. Faithfulness", f"{eval_df['faithfulness'].mean():.2f}")
    with col2:
        st.metric("Avg. Relevance", f"{eval_df['relevance'].mean():.2f}")
    with col3:
        st.metric("Avg. Contextual Precision", f"{eval_df['contextual_precision'].mean():.2f}")
    with col4:
        if 'answer_correctness' in eval_df.columns:
            st.metric("Avg. Answer Correctness", f"{eval_df['answer_correctness'].mean():.2f}")
    
    # Create a radar chart for the average scores
    st.subheader("Average Scores Across All Evaluations")
    metrics = ['faithfulness', 'relevance', 'contextual_precision']
    if 'answer_correctness' in eval_df.columns:
        metrics.append('answer_correctness')
    
    avg_scores = [eval_df[metric].mean() for metric in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar chart
    x = np.arange(len(metrics))
    ax.bar(x, avg_scores, color='skyblue')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Average Evaluation Scores')
    
    # Add score labels on top of bars
    for i, v in enumerate(avg_scores):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    st.pyplot(fig)
    
    # Detailed results table
    st.subheader("Individual Evaluation Results")
    
    # Add a timestamp column in readable format
    eval_df['timestamp'] = pd.to_datetime(eval_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Select columns to display
    display_cols = ['timestamp', 'question'] + metrics + ['average_score']
    st.dataframe(eval_df[display_cols].sort_values('timestamp', ascending=False))
    
    # Option to export results
    if st.button("Export Results to CSV"):
        csv = eval_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

def display_evaluation_results(eval_results):
    """Display evaluation results in the UI."""
    st.success("✅ Evaluation completed!")
    
    # Display metrics with columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Faithfulness", f"{eval_results['faithfulness']:.2f}")
    with col2:
        st.metric("Relevance", f"{eval_results['relevance']:.2f}")
    with col3:
        st.metric("Contextual Precision", f"{eval_results['contextual_precision']:.2f}")
    with col4:
        if 'answer_correctness' in eval_results:
            st.metric("Answer Correctness", f"{eval_results['answer_correctness']:.2f}")
    
    # Display overall score
    st.metric("Overall Score", f"{eval_results['average_score']:.2f}")
    
    # Show retrieved contexts
    with st.expander("View Retrieved Contexts Used for Evaluation"):
        for i, context in enumerate(eval_results['retrieved_contexts']):
            st.markdown(f"**Context {i+1}:**")
            st.text(context)
    
    # Show ground truth comparison
    with st.expander("Compare Answer with Ground Truth"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("System Answer")
            st.write(eval_results['answer'])
        with col2:
            st.subheader("Ground Truth")
            st.write(eval_results['ground_truth'])
    
    # Continue button
    if st.button("Continue Chatting", key="continue_after_eval"):
        st.session_state.evaluation_complete = False
        st.session_state.awaiting_evaluation = False
        st.session_state.current_evaluation_data = None
        st.session_state.evaluation_results_data = None
        st.rerun()

def main():
    """Main application logic."""
    # Streamlit page configuration
    st.set_page_config(page_title="NOC Assist RAG Chatbot", page_icon="🔍", layout="wide")
    
    # Check for viewing evaluation dashboard
    if "view_evaluation" in st.session_state and st.session_state.view_evaluation:
        render_evaluation_dashboard()
        if st.button("Back to Chat"):
            st.session_state.view_evaluation = False
            st.rerun()
        return

    st.title("RAN Ops Assist 🔍📡")
    st.info('Always follow Quality Points', icon="ℹ️")

    # Check for API key
    google_api_key = st.text_input("Enter Gemini API KEY", type="password")
    if not google_api_key:
        st.info("Please add your Google AI API key to continue.", icon="🗝️")
        return

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
        
        # Initialize evaluator
        evaluator = GeminiRagasEvaluator(google_api_key)
        
        # Setup RAG components
        retriever, all_docs = setup_rag_components()
        
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

        # Handle evaluation display if evaluation is complete
        if st.session_state.evaluation_complete and st.session_state.evaluation_results_data:
            st.subheader("✅ Evaluation Results")
            display_evaluation_results(st.session_state.evaluation_results_data)

        # Handle evaluation input if waiting for ground truth
        elif st.session_state.awaiting_evaluation and st.session_state.current_evaluation_data:
            st.subheader("✅ Evaluation")
            with st.form("eval_form"):
                st.write("Please provide the ground truth for this query to evaluate the response:")
                ground_truth = st.text_area("Ground Truth Answer")
                
                submitted = st.form_submit_button("Submit & Evaluate")
                if submitted:
                    with st.spinner("Evaluating..."):
                        # Get stored evaluation data
                        eval_data = st.session_state.current_evaluation_data
                        
                        # Run evaluation
                        eval_results = evaluator.evaluate_rag(
                            question=eval_data["question"],
                            answer=eval_data["answer"],
                            contexts=eval_data["contexts"],
                            ground_truth=ground_truth
                        )
                        
                        # Add metadata
                        eval_results['question'] = eval_data["question"]
                        eval_results['answer'] = eval_data["answer"]
                        eval_results['retrieved_contexts'] = eval_data["contexts"]
                        eval_results['ground_truth'] = ground_truth
                        eval_results['timestamp'] = datetime.now().isoformat()
                        
                        # Store results
                        st.session_state.evaluation_results.append(eval_results)
                        
                        # Update state to show results
                        st.session_state.evaluation_complete = True
                        st.session_state.awaiting_evaluation = False
                        st.session_state.evaluation_results_data = eval_results
                        st.rerun()

        # Handle user input if not in middle of evaluation
        elif not st.session_state.awaiting_evaluation:
            if prompt := st.chat_input("What would you like to know about NOC operations?"):
                # Add user message to chat
                messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                try:
                    with st.spinner("Processing your query..."):
                        # Update chat title if this is the first user message
                        update_chat_title(st.session_state.current_chat_id, messages)
                        
                        # Format chat history
                        chat_history = format_chat_history(messages)
                        
                        # Choose appropriate chain based on question type
                        if is_alarm_related_question(prompt):
                            chain_type = 'alarm'
                        else:
                            chain_type = 'general'
                        
                        response = chains[chain_type].invoke({
                            "input": prompt,
                            "chat_history": chat_history
                        })
                        
                        # Store the retrieved documents for evaluation
                        retrieved_contexts = [doc.page_content for doc in response['context']]
                        
                        # Display assistant response
                        with st.chat_message("assistant"):
                            st.markdown(response['answer'])
                        
                        # Store assistant response
                        messages.append({
                            "role": "assistant", 
                            "content": response['answer']
                        })
                        
                        # If in evaluation mode, prepare for evaluation
                        if st.session_state.evaluation_mode:
                            # Store data for evaluation
                            st.session_state.current_evaluation_data = {
                                "question": prompt,
                                "answer": response['answer'],
                                "contexts": retrieved_contexts
                            }
                            st.session_state.awaiting_evaluation = True
                            st.rerun()

                except Exception as e:
                    st.error(f"An error occurred while generating response: {e}")
                
    except Exception as e:
        st.error(f"Error setting up RAG components: {e}")

if __name__ == "__main__":
    main()