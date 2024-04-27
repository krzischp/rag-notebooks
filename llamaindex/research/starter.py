import logging
import sys
import sys
import os
import openai
from dotenv import load_dotenv, find_dotenv

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Get the current working directory
llamaindex_dir = os.getcwd()
# Get the parent directory
llamaindex_dir = os.path.dirname(llamaindex_dir)

sys.path.append(llamaindex_dir + "/utils")
# sys.path

_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.getenv('OPENAI_API_KEY')

from llamaindex_utils import *

# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader(llamaindex_dir + "/data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
