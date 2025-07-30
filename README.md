<img width="1143" height="243" alt="image" src="https://github.com/user-attachments/assets/226fa48f-c1d1-4a7c-8bb1-7ed853e014a4" />


## Project Overview

This project implements a **personalized recommendation system** using collaborative filtering techniques and provides an interactive **Streamlit web application** for generating recommendations and performing fuzzy searches.  
The system integrates explicit user ratings with implicit bookmarks, applies **matrix factorization (truncated SVD)** to learn latent factors, and enables both recommendation and search functionalities with clickable external links for more details.

<img width="1107" height="570" alt="image" src="https://github.com/user-attachments/assets/9810236c-d1f1-406d-96b3-6d99a487729a" />


## Objectives

- Combine explicit ratings and implicit "to-read" interactions into a unified dataset  
- Filter out users and items with insufficient interactions to reduce sparsity  
- Compute latent factors using truncated SVD for collaborative filtering  
- Recommend similar items to users based on cosine similarity in the latent space  
- Provide fuzzy search functionality for items and authors using RapidFuzz  
- Create a Streamlit interface for recommendations and search, with clickable external links

## Technologies Used

- Python 3.7+  
- Streamlit  
- pandas  
- numpy  
- scipy  
- RapidFuzz  
- scikit-learn (cosine similarity)

<img width="2253" height="1270" alt="image" src="https://github.com/user-attachments/assets/3ef81681-edc8-47f8-a749-d31066db7950" />


## Project Structure

content-recommender-collaborative-filtering/  
├── recommender_system_knn_svd.ipynb        – Jupyter notebook implementing recommendation logic  
├── user_search_url.py                      – Streamlit app for recommendations and search  
├── books_search.py                         – Streamlit module focused on fuzzy search functionality  
├── recsys_assignment_data/                 – Folder containing CSV data files  
│   ├── ratings.csv  
│   ├── to_read.csv  
│   └── books.csv  
├── BTS_Logo.png                            – Image displayed in the Streamlit sidebar  
├── README.md                               – Project documentation  

## Features

- **Data Integration:** Combines explicit ratings with implicit "to-read" data, prioritizing explicit ratings  
- **Data Filtering:** Removes users with <50 interactions and items with <100 interactions  
- **Matrix Factorization:** Computes latent features via truncated SVD for collaborative filtering  
- **Recommendations:** Suggests similar items for a given user using cosine similarity of latent factors  
- **Fuzzy Search:** Uses RapidFuzz for approximate search on item titles and authors  
- **Clickable Links:** Provides external links to detailed pages (e.g., Goodreads) in a new browser tab

<img width="1319" height="724" alt="image" src="https://github.com/user-attachments/assets/446e5ef4-0dd1-460c-ae9f-d9f60e50125e" />


## How It Works

### 1. Data Loading & Preprocessing
- Loads data from ratings.csv, to_read.csv, and books.csv  
- Merges explicit and implicit feedback  
- Filters based on interaction thresholds  

### 2. Matrix Factorization
- Converts the user-item matrix to sparse format  
- Performs truncated SVD to compute latent factors  
- Predicts user preferences based on factorized matrices  

### 3. Recommendation and Search
- Computes cosine similarity to suggest similar items  
- Fuzzy search for titles and authors using RapidFuzz  
- Generates clickable external links for results  

### 4. Streamlit Interface
- **Recommendations Mode:** Enter a user ID to see recommended items  
- **Search Mode:** Search by title or author using fuzzy matching  
- Displays results in tables with clickable links  

## Running the Application

1. Ensure required packages are installed:  
   pip install streamlit pandas numpy scipy rapidfuzz

2. Place the following CSV files inside `recsys_assignment_data/` in the project root:  
   - ratings.csv  
   - to_read.csv  
   - books.csv  

3. Launch the application:  
   streamlit run user_search_url.py  

4. Use the sidebar to switch between **Recommendations** and **Search** modes.

## Bonus: books_search.py Module

The `books_search.py` module is a streamlined version that focuses solely on fuzzy search.  
Key differences:  
- Dedicated UI for search only  
- Additional customization options for fuzzy matching thresholds  
- Optimized for pure search scenarios

## Requirements

pip install streamlit pandas numpy scipy rapidfuzz scikit-learn

## License

MIT License

Copyright (c) 2025 waldepfeifer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
