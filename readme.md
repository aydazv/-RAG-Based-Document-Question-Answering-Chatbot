# 📚 RAG-Based Document Question Answering Chatbot

A Python-based AI chatbot that answers questions about uploaded documents using Retrieval-Augmented Generation (RAG).

The application allows users to upload PDF and TXT files, retrieve relevant text passages through semantic search, and generate answers using a local language model.

## ✨ Features

- 📄 **Document Processing:** Extract text from PDF and TXT files.
- ✂️ **Text Chunking:** Split documents into smaller, overlapping text segments.
- 🔎 **Semantic Search:** Retrieve relevant passages using multilingual text embeddings.
- 🤖 **AI-Powered Answers:** Generate answers based on retrieved document content.
- 💬 **Interactive Chat Interface:** Ask questions through a Streamlit web application.
- 📌 **Source Tracking:** Display source filenames, chunk numbers, and similarity scores.
- 🔒 **Local Model Inference:** Run embedding and language models locally without requiring a paid LLM API.

## 🛠️ Technologies Used

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- Qwen2.5-0.5B-Instruct
- Multilingual E5 Small
- PyPDF

## ⚙️ How It Works

The application follows the Retrieval-Augmented Generation (RAG) pipeline:

1. **Document Upload:** Users upload PDF or TXT files.
2. **Text Extraction:** Text is extracted from the uploaded documents.
3. **Chunking:** Extracted text is divided into smaller overlapping chunks.
4. **Embedding Generation:** Each chunk is converted into a numerical vector.
5. **Semantic Retrieval:** The user's question is embedded and compared with document vectors.
6. **Context Construction:** The most relevant passages are selected and combined.
7. **Answer Generation:** A language model generates a response using the retrieved context.
8. **Source Display:** The application displays the retrieved document sources.

## 📂 Project Structure

```text
RAG-Chatbot/
│
├── documents/
│   └── machine_learning.txt
│
├── main.py
├── app.py
├── test_llm.py
├── README.md
└── .gitignore
```

- `main.py`: Document reading, chunking, semantic retrieval, and command-line RAG implementation.
- `app.py`: Streamlit web application.
- `test_llm.py`: Script for testing the language model.
- `documents/`: Sample documents used for testing.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RAG-Chatbot.git
cd RAG-Chatbot
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -U pip
python -m pip install streamlit pypdf sentence-transformers transformers accelerate
```

PyTorch will be installed as a dependency when required. If installation fails, install a compatible PyTorch version for your operating system.

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

## 🧪 Usage

1. Launch the Streamlit application.
2. Upload one or more PDF or TXT documents.
3. Click **Process Documents**.
4. Enter a question related to the uploaded documents.
5. Read the generated answer.
6. Expand the source section to inspect the retrieved passages.

## 🧠 Models

**Embedding Model:** `intfloat/multilingual-e5-small`

Used to generate multilingual text embeddings for semantic retrieval.

**Language Model:** `Qwen/Qwen2.5-0.5B-Instruct`

Used to generate natural-language answers from retrieved document passages.

Both models are downloaded from Hugging Face when first loaded and run locally.

## ⚠️ Limitations

- Answer quality depends on document quality, retrieval accuracy, and model capabilities.
- The small language model may produce incomplete or inaccurate answers.
- Similarity scores are not probabilities or guarantees of correctness.
- Scanned PDFs may require OCR, which is not currently implemented.
- Large documents and models may require significant RAM and processing time.
- Models require an internet connection for their initial download unless already cached.

## 🔮 Future Improvements

- [ ] Add persistent vector storage.
- [ ] Improve retrieval with configurable similarity thresholds.
- [ ] Support additional document formats.
- [ ] Add OCR for scanned PDFs.
- [ ] Display more precise citations, including PDF page numbers.
- [ ] Add conversation memory.
- [ ] Improve the user interface and error handling.
- [ ] Add automated tests and evaluation metrics.
- [ ] Deploy the application for remote access.

## 👩‍💻 Author

**Aida Zivari Dana**

Computer Science Graduate | Python | Machine Learning | Natural Language Processing

## 📄 License

This project is available under the MIT License. If you choose this license, add a `LICENSE` file containing the MIT License text before publishing the repository.