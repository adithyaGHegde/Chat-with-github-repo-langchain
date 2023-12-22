# Chat-with-github-repo-langchain
Submission for the Full stack AI assignment for AI planet! A chatbot that let's you talk to your Github repository!
This is an end to end LLM project that uses Langchain and Streamlit to host LLM model to allow you to chat with your documents or with a Github repository!

We will build an LLM based question and answer system that will use following,
  * Orca Mini 3b model, or any other model
  * Hugging face embeddings
  * Streamlit for UI
  * Langchain framework
  * Chromadb as a vector store

## Setup
1.Clone this repository to your local machine using:
```bash
  git clone https://github.com/adithyaGHegde/Chat-with-github-repo-langchain.git
```

2.Navigate to the project directory:
```bash
  cd Chat-with-github-repo-langchain
```

3. Install the requirements
```bash
pip install -r requirements.txt
```

4. Set your HUGGINGFACEHUB_API_TOKEN in the .env file, or your Google API key

5. In a ./models folder, make sure the GGML model file is present before running
```bash
streamlit run app.py
```

The application will now be running on your browser. In the sidebar add documents/repo link and click on Process. Wait until its done, and then you can chat with the bot!
