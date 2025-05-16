# Build with AI - RAG Event Bot

This project is a **Streamlit-based chatbot** that uses Google Gemini LLM and ChromaDB vector store to answer questions about uploaded files (PDF, TXT, DOCX) or event information. It supports **dynamic context**: when you upload a new file, only that file's content is used for answering questions.

---

## Features

- **Chatbot UI** with Streamlit
- **File upload** (PDF, TXT, DOCX) — ask questions about your uploaded file
- **RAG (Retrieval Augmented Generation)** using ChromaDB and Google Gemini
- **Dynamic context**: Only the latest uploaded file is used (old data is deleted)
- **Custom prompt**: If you upload a resume, the bot acts as a Resume Assistant; otherwise, it acts as an Event Bot
- **Hindi/English friendly**

---

## Getting Started

### 1. Clone the repository

```sh
git clone <repo-url>
cd Event_bot_RAG_ChromaDB_demo
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

### 3. Get a Free Gemini API Key from Google AI Studio

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click on **"Create API Key"**.
4. Copy the generated API key.

> **Note:** Free Gemini API keys have usage limits. For more, see [Google Gemini API documentation](https://ai.google.dev/).

### 4. Add your Google Gemini API key

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=your_actual_api_key_here
```

### 5. Run the app locally

```sh
streamlit run app.py
```

Open the URL shown in your terminal (usually http://localhost:8501).

---

## Usage

1. **Upload a file** (PDF, TXT, DOCX) using the upload widget.
2. **Ask questions** about the content of your uploaded file in the chat box.
3. If you upload a resume (file name contains `resume`, `cv`, or is a PDF), the bot will act as a Resume Assistant.
4. If you upload any other document, the bot will answer based on that file's content.
5. **Only the latest uploaded file is used** for answering questions. Old data is automatically deleted.

---

## Deploy on Streamlit Cloud

You can deploy this app for free on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push your code to a public GitHub repository.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in.
3. Click **"New app"** and select your repository and branch.
4. Set `app.py` as the main file.
5. In the **"Advanced settings"**, add a secret called `GEMINI_API_KEY` with your Gemini API key.
6. Click **"Deploy"**.

> **Note:** On Streamlit Cloud, use the **Secrets** feature for your API key instead of a `.env` file.

---

## Notes

- If you ask something not present in the uploaded file, the bot will reply:  
  **"File me aisi jankari nahi hai."**
- If no file is uploaded, the bot will act as an Event Bot (if you enable a default event context).
- For PDF support, `PyPDF2` is used. For DOCX, `python-docx` is required.

---

## Troubleshooting

- **streamlit: command not found**  
  → Run `pip install streamlit`
- **API key not found**  
  → Make sure `.env` file is present and contains your `GEMINI_API_KEY`
- **Chroma directory not found**  
  → The app will create the `chroma` directory automatically on first run.

---

## License

MIT

---

**Made with ❤️ using Streamlit, Google Gemini, and ChromaDB**