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
    Compute truncated SVD on the user-book matrix and return the predicted ratings.
    """
    sparse_matrix = csr_matrix(matrix.values)
    U, sigma, Vt = svds(sparse_matrix, k=k)
    sigma = np.diag(sigma)
    # Reverse order to get descending singular values.
    U = U[:, ::-1]
    sigma = sigma[::-1, ::-1]
    Vt = Vt[::-1, :]
    R_pred_combined = np.dot(np.dot(U, sigma), Vt)
    return R_pred_combined

@st.cache_resource
def create_books_mapping(books: pd.DataFrame):
    """
    Create a dictionary mapping book_id to details (title, authors, goodreads_book_id)
    and prepare the fuzzy search index.
    """
    # Build mapping for quick lookup.
    books_map = books.set_index('book_id')[['title', 'authors', 'goodreads_book_id']].to_dict('index')
    # Build a search index: combine title and authors (in lower-case).
    books = books.copy()
    books['search_str'] = (books['title'].fillna('') + " " + books['authors'].fillna('')).str.lower()
    search_choices = books['search_str'].tolist()
    book_ids_list = books['book_id'].tolist()
    return books_map, search_choices, book_ids_list

# 2. Recommendation & Fuzzy Search Functions

def recommend_books(user_id, R_pred, original_matrix, books_map, top_n=5):
    """
    For a given user_id, return a DataFrame with the top_n recommended books.
    The goodreads link will be clickable.
    """
    if user_id not in original_matrix.index:
        return pd.DataFrame()
    user_idx = original_matrix.index.get_loc(user_id)
    user_pred = R_pred[user_idx, :]
    # Exclude books already rated/bookmarked.
    rated_books = set(original_matrix.columns[original_matrix.loc[user_id] > 0])
    preds_df = pd.DataFrame({
        'book_id': original_matrix.columns,
        'predicted_rating': user_pred
    })
    preds_df = preds_df[~preds_df['book_id'].isin(rated_books)]
    preds_df = preds_df.sort_values(by='predicted_rating', ascending=False).head(top_n)
    # Append book details.
    preds_df['title'] = preds_df['book_id'].apply(lambda x: books_map.get(x, {}).get('title', 'Unknown Title'))
    preds_df['authors'] = preds_df['book_id'].apply(lambda x: books_map.get(x, {}).get('authors', 'Unknown Author'))
    # Create a clickable goodreads link using the goodreads_book_id.
    preds_df['goodreads_link'] = preds_df['book_id'].apply(
        lambda x: f'<a href="https://www.goodreads.com/book/show/{books_map.get(x, {}).get("goodreads_book_id", "")}" target="_blank">View on Goodreads</a>'
    )
    preds_df['user_id'] = user_id
    return preds_df[['user_id', 'book_id', 'title', 'authors', 'goodreads_link', 'predicted_rating']]

def fuzzy_search(query, books_map, search_choices, book_ids_list, limit=10):
    """
    Perform fuzzy search on book titles and authors using RapidFuzz.
    The goodreads link will be clickable.
    """
    query = query.lower()
    results = process.extract(query, search_choices, scorer=fuzz.token_sort_ratio, limit=limit)
    matches = []
    for match, score, idx in results:
        book_id = book_ids_list[idx]
        details = books_map.get(book_id, {})
        goodreads_link = f'<a href="https://www.goodreads.com/book/show/{details.get("goodreads_book_id", "")}" target="_blank">View on Goodreads</a>'
        matches.append({
            'book_id': book_id,
            'title': details.get('title', 'Unknown Title'),
            'authors': details.get('authors', 'Unknown Author'),
            'goodreads_link': goodreads_link,
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

# Compute SVD and predicted ratings
R_pred_combined = compute_svd(combined_user_book_filled, k=50)

# Build the books mapping and fuzzy search index
books_map, search_choices, book_ids_list = create_books_mapping(books)

# Sidebar Navigation
st.sidebar.title("Navigation")

if app_mode == "Recommendations":
    st.header("Get Book Recommendations")
    st.markdown("Enter a user ID to get personalized book recommendations.")
    
    user_id_input = st.text_input("User ID", value=str(combined_user_book_filled.index[0]))
    top_n = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)
    
    if st.button("Get Recommendations"):
        try:
            user_id = int(user_id_input)
            recs_df = recommend_books(user_id, R_pred_combined, combined_user_book_filled, books_map, top_n=top_n)
            if recs_df.empty:
                st.error("User not found or no recommendations available.")
            else:
                st.subheader(f"Recommendations for User {user_id}")
                # Display the DataFrame as HTML to render clickable links.
                st.markdown(recs_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        except ValueError:
            st.error("Please enter a valid numeric user ID.")

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
                st.markdown(results_df.to_html(escape=False, index=False), unsafe_allow_html=True)

st.markdown("Developed using Streamlit, SciPy, and RapidFuzz.")