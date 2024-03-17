# Retrieval-Augmented-Generation-Chatbot:
---

Customizable chatbot which will read PDFs and answer your questions for you. ChatGPT but free!
Built with chainlit and langchain.

For best results, install llama-cpp-python with GPU support.

## Steps to use:
    1. Install requirements with `pip install -r requirements.txt`
    2. Place pdf documents into data directory (defaults to `./data`)
    3. Run the app with python (Usage guidance below)
    4. Interact with the chatbot at the port that chainlit runs on (8000 by default)

For the simplest running experience, just drop pdfs into './data' and run `python app.py -d`, which will download Llama-2-7B-Chat-GGUF for you (https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q2_K.gguf).

## Usage

```commandline
usage: RAGChatBot [-h] [-data_dir DATA_DIR] [-model_path MODEL_PATH] [-d]

Retrieval Augmented Generation Chatbot

optional arguments:
  -h, --help            show this help message and exit
  -data_dir DATA_DIR    Path to directory contianing all pdfs.
  -model_path MODEL_PATH
                        Path to LLM model with .gguf extension.
  -d                    Flag to download user's requested model, defaults to TheBloke/Llama-2-7B-Chat-GGUF
```
### Samples Usage 
```commandline
python app.py --data_dir "./data" --model_path "./model/llama-2-7b-chat.Q2_K.gguf"
```

### Directory Tree
~~~
Retrieval-Augmented-Generation-Chatbot
    └── model
        └── .gguf model
    └── data
        └── Document1.pdf
        ├── Document2.pdf
        └── Document3.pdf
~~~




