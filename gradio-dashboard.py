import pandas as pd
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma # opensource vector database
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

# load the cleaned df
books = pd.read_csv('./data/books_with_emotion_scores.csv')

# I'll use the thumbnail for the display
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isnull(), "cover-not-found.jpg", books["large_thumbnail"])

raw_documents = TextLoader('./data/tagged_descriptions.txt', encoding='utf-8').load()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
embedding = MistralAIEmbeddings(
    model="mistral-embed",
    api_key= "7pe3roabiqOZV3bUCcVsevNwX3D3pyxL"
)
db_books = Chroma.from_documents(documents, embedding)

# semantic recommendation retrival
def retrieve_semantic_recommendation(
        query: str,
        category: str = None,
        tone: str = None,
        initia_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initia_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    filtered_books = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category != "All":
        filtered_books = filtered_books[filtered_books['simple_categories'] == category][:final_top_k]
    else:
        filtered_books = filtered_books.head(final_top_k)

    if tone == "Happy":
       filtered_books.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Sad":
         filtered_books.sort_values(by="sadness", ascending=False, inplace=True)
    elif tone == "Angry":
         filtered_books.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
         filtered_books.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Surprising":
         filtered_books.sort_values(by="surprise", ascending=False, inplace=True)

    return filtered_books


def recommend_books(query: str, category: str, tone: str):
    recommended_books = retrieve_semantic_recommendation(query, category, tone)
    results = []

    for _, row in recommended_books.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:50]) + "..."
        authors_split = row["authors"].split(",")

        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]}, {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:1])} and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

categories = ['All'] + sorted(books['simple_categories'].unique())
tones = ["All"] + ["Happy", "Sad", "Angry", "Suspenseful", "Surprising"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# BayaBooks Recommendation Dashboard")
    gr.Markdown("Get book recommendations based on your interests and mood!")

    with gr.Row():
        user_query = gr.Textbox(placeholder="e.g., A Story about success", label="Please enter a description of what you're looking for:", lines=3)
        category_dropdown = gr.Dropdown(choices=categories, value="All", label="Select Book Category:")
        tone_dropdown = gr.Dropdown(choices=tones, value="All", label="Select Desired Tone/Mood:")
        submit_button = gr.Button("Get Recommendations")

        gr.Markdown("### Recommended Books:")
        output_gallery = gr.Gallery(label="Recommended Books", columns = 8, rows = 2, elem_id="gallery")

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output_gallery
    )

if __name__ == "__main__":
    dashboard.launch(share=True)