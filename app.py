import os
import argparse 
import json

from huggingface_hub import hf_hub_download

def init(args):
    
    model_path = args.model_path
    data_path = args.data_dir
    
    # download:
    if args.d:
        print("Downloading default model into provided model path now...")
        if not os.path.exists(model_path):
            default_model="TheBloke/Llama-2-7B-Chat-GGUF"
            hf_hub_download(repo_id=default_model, filename="llama-2-7b-chat.Q2_K.gguf", local_dir=model_path)
        
    # verify that model is downloaded
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Model Path {}does not exist!".format(model_path))
    
    # verify data
    if not os.path.exists(data_path):
        raise FileNotFoundError("Data Directory {} does not exist! Ensure that you create it and place pdfs into it.".format(data_path))
    elif len(os.listdir(data_path)) == 0:
        raise RuntimeError('Data Directory {} is empty'.format(data_path))
    

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                        prog='RAGChatBot',
                        description="Retrieval Augmented Generation Chatbot",
                        )
    parser.add_argument("-data_dir", type=str, default="./data", help="Path to directory contianing all pdfs.")
    parser.add_argument("-model_path", type=str, default="./model/llama-2-7b-chat.Q2_K.gguf", help="Path to LLM model with .gguf extension.")
    parser.add_argument("-d", action='store_true', help="Flag to download user's requested model, defaults to TheBloke/Llama-2-7B-Chat-GGUF")
    args = parser.parse_args()
    
    init(args)
    arg_dict = vars(args)
    
    # save args
    with open("args.json", 'w') as f:
        f.write(json.dumps(arg_dict))
            
    os.system("chainlit run clrun.py")
    