# 📡 RAN Ops Assist — AI-powered NOC Assistant with RAG

Welcome to **RAN Ops Assist**, your AI-powered Network Operations Center (NOC) assistant specifically tailored for **Radio Access Network (RAN)** operations.

This Streamlit app leverages **LangChain**, **Google Gemini**, **FAISS**, and **PDF document ingestion** to provide real-time, context-aware, structured assistance for telecom engineers.

---

## 🚀 Features

- 💬 Conversational interface with structured telecom troubleshooting responses
- 📚 RAG (Retrieval-Augmented Generation) using PDF-based knowledge base
- 🧠 Multilingual support: English, Romanian, German
- 🔍 Smart intent routing between **Alarm-based** and **General** question handling
- 📂 Persistent chat history with titles and timestamps
- 🔐 Secure API key entry (Google Gemini)

---

## 🧱 Tech Stack

- **Frontend:** Streamlit
- **LLM:** Google Gemini (via `langchain-google-genai`)
- **RAG:** LangChain + FAISS + HuggingFace Embeddings
- **File Loader:** PDF ingestion via `PyPDFDirectoryLoader`
- **Languages:** Python 3.10+

---

## 📂 Directory Structure

```
📁 your_project/
├── 📄 app.py                # Main Streamlit app file
├── 📁 pdf files/            # Directory containing PDFs for RAG
├── 📄 .env                  # Store your environment variables (like LOGO_PATH)
|-- 📄 app_with_eval.py      # Main app with evaluation feature
```

---

## 🔧 Setup Instructions

1. **Clone the Repository:**

```bash
git clone https://github.com/yourusername/ran-ops-poc.git
cd ran-ops-poc
```

2. **Create a virtual environment and activate it:**

```bash
conda create -p myenv python==3.10 -y
conda activate myenv/  # On Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Add Environment Variables in `.env`**


> 💡 _The Gemini API Key is entered manually within the app UI._

5. **Add your reference PDFs**

Place all your technical documents into the `pdf files/` directory.

6. **Run the App**

```bash
streamlit run app.py
```

---

## 🔑 Gemini API Key

To use this app, you need access to **Google Gemini API**:

- Get your key from [Google AI Studio](https://makersuite.google.com/app)
- Paste it into the app when prompted.

---

## 🌐 Supported Languages

- **English**
- **Romanian**
- **German**

You can change the language from the sidebar at any time.

---

## 🧠 RAG Behavior

| Type                     | Behavior                                                                 |
|--------------------------|--------------------------------------------------------------------------|
| **Alarm/Technical Query**| Uses structured response template including cause, recommendation, SOPs |
| **General/History Query**| Conversational, telecom-contextual responses only                        |

---

## 📌 Example Use Cases

- "How to handle No Connection to Unit"
- "How to handle Failure in Optical Interface?"
- "What did I ask last time?"
- "What's the root cause of high RF interference?"

---

## 🤖 Future Enhancements

- 🔄 Chat export and persistent memory
- 🌐 Integration with live NOC databases and ticketing systems
- 📊 Analytics on chat and issue trends

---

## 🙏 Credits

- Built using [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io/), and [Google Gemini](https://ai.google.dev/).
- Embeddings by HuggingFace.
- Vector Store powered by FAISS.

---

## 📜 License

MIT License. Feel free to fork and enhance.

---
