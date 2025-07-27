import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from rapidfuzz import process, fuzz

# 1. Data Loading & Caching

@st.cache_resource
def load_raw_data():
    """Load raw CSV data just once as a resource."""
    ratings = pd.read_csv("recsys_assignment_data/ratings.csv")
    to_read = pd.read_csv("recsys_assignment_data/to_read.csv")
    books = pd.read_csv("recsys_assignment_data/books.csv")
    return ratings, to_read, books

@st.cache_resource
def preprocess_data(ratings: pd.DataFrame, to_read: pd.DataFrame):
    """
    Combine explicit ratings with implicit "to-read" data,
    filter users/books with low interactions, and create
    a combined user-book matrix.
    """
    # Assign an implicit rating to "to read" entries.
    implicit_rating_value = 4.0
    to_read = to_read.copy()
    to_read['rating'] = implicit_rating_value

    explicit_ratings = ratings.copy()
    # Combine explicit ratings (taking precedence) with implicit data.
    explicit_ratings_indexed = explicit_ratings.set_index(['user_id', 'book_id'])
    to_read_indexed = to_read.set_index(['user_id', 'book_id'])
    combined_ratings = explicit_ratings_indexed.combine_first(to_read_indexed).reset_index()
    
    # Filter out users and books with very few interactions.
    min_user_entries = 50
    min_book_entries = 100
    user_counts = combined_ratings.groupby('user_id').size()
    book_counts = combined_ratings.groupby('book_id').size()
    valid_users = user_counts[user_counts >= min_user_entries].index
    valid_books = book_counts[book_counts >= min_book_entries].index
    combined_ratings_filtered = combined_ratings[
        (combined_ratings.user_id.isin(valid_users)) &
        (combined_ratings.book_id.isin(valid_books))
    ].copy()
    
    # Create the combined user-book matrix.
    combined_user_book = combined_ratings_filtered.pivot(index='user_id', columns='book_id', values='rating')
    combined_user_book_filled = combined_user_book.fillna(0)
    
    return combined_ratings_filtered, combined_user_book_filled

@st.cache_resource
def compute_svd(matrix: pd.DataFrame, k=50):
    """
    Compute truncated SVD on the user-book matrix and return the predicted ratings and SVD factors.
    """
    sparse_matrix = csr_matrix(matrix.values)
    U, sigma, Vt = svds(sparse_matrix, k=k)
    sigma = np.diag(sigma)
    # Reverse order to get descending singular values.
    U = U[:, ::-1]
    sigma = sigma[::-1, ::-1]
    Vt = Vt[::-1, :]
    R_pred_combined = np.dot(np.dot(U, sigma), Vt)
    return R_pred_combined, U, sigma, Vt

@st.cache_resource
def create_books_mapping(books: pd.DataFrame):
    """
    Create a dictionary mapping book_id to details (title and authors)
    and prepare the fuzzy search index.
    """
    # Build mapping for quick lookup.
    books_map = books.set_index('book_id')[['title', 'authors']].to_dict('index')
    # Build a search index: combine title and authors (in lower-case).
    books = books.copy()
    books['search_str'] = (books['title'].fillna('') + " " + books['authors'].fillna('')).str.lower()
    search_choices = books['search_str'].tolist()
    book_ids_list = books['book_id'].tolist()
    return books_map, search_choices, book_ids_list

# 2. Recommendation & Fuzzy Search Functions

def recommend_books_by_book(book_id, Vt, original_matrix, books_map, top_n=5):
    """
    For a given book_id, return a DataFrame with the top_n similar books based on latent factors.
    """
    # Ensure the book exists in our combined matrix (columns represent book_ids).
    if book_id not in original_matrix.columns:
        return pd.DataFrame()
    
    # Compute latent book factors: Vt is (k x num_books), so transpose it.
    book_factors = Vt.T  # shape: (num_books, k)
    # Map column order to book_ids.
    all_book_ids = list(original_matrix.columns)
    
    # Get the index of the query book.
    idx = all_book_ids.index(book_id)
    target_vector = book_factors[idx]
    
    # Compute cosine similarities in a vectorized manner.
    # Dot product between target_vector and all book factors.
    dot_products = book_factors.dot(target_vector)
    # Norms for all book vectors and the target vector.
    norms = np.linalg.norm(book_factors, axis=1)
    target_norm = np.linalg.norm(target_vector)
    cosine_sim = dot_products / (norms * target_norm + 1e-10)
    
    # Create a DataFrame with similarity scores.
    sim_df = pd.DataFrame({
        'book_id': all_book_ids,
        'similarity': cosine_sim
    })
    # Exclude the book itself.
    sim_df = sim_df[sim_df['book_id'] != book_id]
    sim_df = sim_df.sort_values(by='similarity', ascending=False).head(top_n)
    
    # Append book details using vectorized mapping.
    sim_df['title'] = sim_df['book_id'].map(lambda x: books_map.get(x, {}).get('title', 'Unknown Title'))
    sim_df['authors'] = sim_df['book_id'].map(lambda x: books_map.get(x, {}).get('authors', 'Unknown Author'))
    sim_df['query_book_id'] = book_id
    return sim_df[['query_book_id', 'book_id', 'title', 'authors', 'similarity']]

def fuzzy_search(query, books_map, search_choices, book_ids_list, limit=10):
    """
    Perform fuzzy search on book titles and authors using RapidFuzz.
    """
    query = query.lower()
    results = process.extract(query, search_choices, scorer=fuzz.token_sort_ratio, limit=limit)
    matches = []
    for match, score, idx in results:
        b_id = book_ids_list[idx]
        details = books_map.get(b_id, {})
        matches.append({
            'book_id': b_id,
            'title': details.get('title', 'Unknown Title'),
            'authors': details.get('authors', 'Unknown Author'),
            'match_score': score
        })
    return pd.DataFrame(matches)

# 3. Streamlit UI

# Example usage: Display an image in the sidebar
st.sidebar.image("BTS_Logo.png", use_column_width=True)

app_mode = st.sidebar.selectbox("Choose the app mode", ["Recommendations", "Search Books/Authors"])

st.title("Book Recommender System")
st.markdown("Deploying a recommender system with fuzzy search for books and authors.")

# Load raw data once as a resource
ratings, to_read, books = load_raw_data()

# Preprocess data: combine explicit and implicit ratings
combined_ratings_filtered, combined_user_book_filled = preprocess_data(ratings, to_read)

# Compute SVD and predicted ratings, and retrieve SVD factors.
R_pred_combined, U, sigma, Vt = compute_svd(combined_user_book_filled, k=50)

# Build the books mapping and fuzzy search index
books_map, search_choices, book_ids_list = create_books_mapping(books)

# Sidebar Navigation
st.sidebar.title("Navigation")

if app_mode == "Recommendations":
    st.header("Get Similar Books")
    st.markdown("Enter a book ID to see similar book recommendations based on latent features.")
    
    book_id_input = st.text_input("Book ID", value=str(list(combined_user_book_filled.columns)[0]))
    top_n = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)
    
    if st.button("Get Recommendations"):
        try:
            # Convert input to the appropriate type.
            book_id = int(book_id_input)
            recs_df = recommend_books_by_book(book_id, Vt, combined_user_book_filled, books_map, top_n=top_n)
            if recs_df.empty:
                st.error("Book not found or no recommendations available.")
            else:
                st.subheader(f"Books similar to Book ID {book_id}")
                st.table(recs_df)
        except ValueError:
            st.error("Please enter a valid numeric book ID.")

elif app_mode == "Search Books/Authors":
    st.header("Fuzzy Search Books & Authors")
    st.markdown("Search for books and authors using approximate matching. For example, try 'Scot fitzgerad' to find 'F. Scott Fitzgerald'.")
    
    query = st.text_input("Enter your search query")
    if st.button("Search"):
        if query.strip() == "":
            st.error("Please enter a non-empty query.")
        else:
            results_df = fuzzy_search(query, books_map, search_choices, book_ids_list, limit=10)
            if results_df.empty:
                st.info("No matches found.")
            else:
                st.subheader("Search Results")
                st.table(results_df)

st.markdown("Developed using Streamlit, SciPy, and RapidFuzz.")
