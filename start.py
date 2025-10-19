from dotenv import load_dotenv

from utils.embeding_utils import main_pipeline_embedding_process_pdfs_to_pgvector, search_similar_chunks


# Load environment variables from .env file
load_dotenv()

start = True
search_text_input = "what is deepseek"

def process_the_files():
    main_pipeline_embedding_process_pdfs_to_pgvector()

def verify():
    result = search_similar_chunks(search_text_input)
    print(result)
    

# Test your knowledge-powered agent
if __name__ == "__main__":
    if start : 
        process_the_files()
    else :
        verify()
