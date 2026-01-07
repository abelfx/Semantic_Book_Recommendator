## Semantic Book Recommendator

A content-based semantic book recommendation system that generates personalized book suggestions based on user-provided textual descriptions of their reading preferences. The system leverages NLP techniques, vector similarity search, and sentiment analysis to improve book discovery.

### Features

- Semantic Search using vector embeddings
- Text-based recommendations from user-described interests
- Sentiment Analysis on book descriptions
- Exploratory Data Analysis and preprocessing
- Interactive Gradio Dashboard for user-friendly interaction
- Modular and extensible design for future ML/AI enhancements

Project Structure
```
Semantic_Book_Recommendator/
│
├── data-exploration.ipynb      # Data cleaning, analysis, and preprocessing
├── vector_search.ipynb         # Semantic vector search implementation
├── sentiment-analysis.ipynb    # Sentiment analysis on book data
├── gradio-dashboard.py         # Interactive Gradio UI for recommendations
├── cover-not-found.jpg         # Fallback book cover image
├── README.md                   # Project documentation
├── .env                        # Environment variables
└── .gitignore                  # Ignored files
```

### How It Works

Users provide a natural language description of the books they like (themes, genres, tone, style).
The system processes the input using text embeddings.
Book descriptions are converted into vectors and compared using semantic similarity.
The most relevant books are returned as recommendations.
Results are displayed through an interactive Gradio interface.

### Technologies Used

- Python
- Natural Language Processing (NLP)
- Vector Embeddings & Similarity Search
- Gradio
- Pandas, NumPy
- Jupyter Notebook

## Running the Project
1️⃣ Install dependencies
```
pip install -r requirements.txt
```

2️⃣ Run the Gradio dashboard
```
python gradio-dashboard.py
```

3️⃣ Open the provided local URL in your browser
🎯 Use Cases

- Personalized book discovery platforms
- Educational recommendation systems
- NLP and semantic search demonstrations
- Portfolio project for AI / Software Engineering roles

### 📌 Future Improvements

Integrate advanced transformer-based embeddings
Add collaborative filtering
Deploy as a web service
Improve recommendation ranking with user feedback

### 👤 Author

- Abel Tesfa
- Software Engineering Student
- GitHub: abelfx

