import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma # opensource vector database
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

# load the cleaned df
books = pd.read_csv('books_with_emotion_scores.csv')

# I'll use the thumbnail for the display
books["large_thumbnail"] = books["large_thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isnull(), "cover-not-found.jpg", books["large_thumbnail"])