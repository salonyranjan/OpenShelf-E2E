import os
import sys
import pickle
import streamlit as st
import numpy as np
import random
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException

# --- PAGE CONFIG ---
st.set_page_config(page_title="OpenShelf | Horizon", layout="wide", initial_sidebar_state="collapsed")

class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    @st.cache_resource
    def _load_pickles(_self):
        book_pivot = pickle.load(open(_self.recommendation_config.book_pivot_serialized_objects, 'rb'))
        final_rating = pickle.load(open(_self.recommendation_config.final_rating_serialized_objects, 'rb'))
        model = pickle.load(open(_self.recommendation_config.trained_model_path, 'rb'))
        return book_pivot, final_rating, model

    def recommend_book(self, book_name, n=6):
        try:
            book_pivot, final_rating, model = self._load_pickles()
            book_id = np.where(book_pivot.index == book_name)[0][0]
            _, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=n)
            
            names = [book_pivot.index[idx] for idx in suggestion[0]]
            posters = [final_rating[final_rating['title'] == name]['image_url'].iloc[0] for name in names]
            return names, posters
        except Exception as e:
            raise AppException(e, sys) from e

# --- HORIZON UI THEME ---
def apply_horizon_theme():
    st.markdown("""
        <style>
        /* Deep Space Background */
        .stApp {
            background: linear-gradient(180deg, #050505 0%, #1a1a2e 100%);
            color: #ffffff;
        }
        
        /* Floating Top Nav */
        .top-nav {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            padding: 15px 50px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 0 0 20px 20px;
            margin-bottom: 30px;
        }

        /* Feature Cards */
        .stat-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }

        /* Modern Glow Button */
        .stButton>button {
            background: linear-gradient(90deg, #ff00cc 0%, #3333ff 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px 30px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            box-shadow: 0 0 20px rgba(255, 0, 204, 0.6);
            transform: scale(1.05);
            color: white;
        }

        /* Book Item View */
        .book-item {
            text-align: center;
            padding: 10px;
        }
        .book-item img {
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            transition: 0.4s;
        }
        .book-item img:hover {
            transform: translateY(-10px) rotate(2deg);
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    apply_horizon_theme()
    obj = Recommendation()
    
    # --- TOP NAVIGATION BAR ---
    st.markdown("""
        <div class="top-nav">
            <h2 style='margin:0; color:#ff00cc;'>OpenShelf</h2>
            <div style='display:flex; gap:20px; color:#888;'>
                <span>Collaborative Filtering</span>
                <span>•</span>
                <span>AI-Powered Recommendations</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- HERO SECTION ---
    book_names_path = os.path.join('templates', 'book_names.pkl')
    book_names = pickle.load(open(book_names_path, 'rb'))

    # Stats Ribbon (NEW FEATURE)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="stat-box"><h4>📚 Library Size</h4><h2 style="color:#ff00cc;">'+str(len(book_names))+'</h2></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-box"><h4>⚡ Algorithm</h4><h2 style="color:#3333ff;">K-Nearest</h2></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-box"><h4>🌲 Growth</h4><h2 style="color:#00ff88;">Active</h2></div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- SEARCH EXPERIENCE ---
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        selected_book = st.selectbox("Search for a book title...", book_names, label_visibility="collapsed")
    with col_btn:
        search_btn = st.button("Unveil Discovery ✨")

    # --- RESULTS ---
    if search_btn:
        with st.spinner("Analyzing neural reader patterns..."):
            names, posters = obj.recommend_book(selected_book)
            
            st.markdown(f"### Results for: <span style='color:#ff00cc;'>{selected_book}</span>", unsafe_allow_html=True)
            st.markdown("---")
            
            cols = st.columns(5)
            # Displaying indices 1-5 (Skipping searched book at index 0)
            for i in range(1, 6):
                with cols[i-1]:
                    st.markdown(f"""
                        <div class="book-item">
                            <img src="{posters[i]}" style="width:100%;">
                            <p style="font-size:12px; margin-top:10px; font-weight:bold; color:#ccc;">{names[i]}</p>
                        </div>
                    """, unsafe_allow_html=True)

    # --- FOOTER / ADMIN ---
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    if st.button("🔄 Sync Library & Re-train Model"):
        with st.spinner("Planting new seeds in the matrix..."):
            TrainingPipeline().start_training_pipeline()
            st.success("The library has bloomed! 🌳")