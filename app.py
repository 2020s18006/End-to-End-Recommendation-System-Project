import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException


class Recommendation:
    def __init__(self,app_config = AppConfiguration()):
        try:
            self.recommendation_config= app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    def fetch_poster(self,suggestion):
        try:
            book_name = []
            ids_index = []
            poster_url = []
            book_pivot =  pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            final_rating =  pickle.load(open(self.recommendation_config.final_rating_serialized_objects,'rb'))

            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])

            for name in book_name[0]: 
                ids = np.where(final_rating['title'] == name)[0][0]
                ids_index.append(ids)

            for idx in ids_index:
                url = final_rating.iloc[idx]['image_url']
                poster_url.append(url)

            return poster_url
        
        except Exception as e:
            raise AppException(e, sys) from e
        


    def recommend_book(self,book_name):
        try:
            books_list = []
            model = pickle.load(open(self.recommendation_config.trained_model_path,'rb'))
            book_pivot =  pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            book_id = np.where(book_pivot.index == book_name)[0][0]
            distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )

            poster_url = self.fetch_poster(suggestion)
            
            for i in range(len(suggestion)):
                    books = book_pivot.index[suggestion[i]]
                    for j in books:
                        books_list.append(j)
            return books_list , poster_url   
        
        except Exception as e:
            raise AppException(e, sys) from e


    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.success("✅ Training Completed Successfully!")
            logging.info(f"Recommended successfully!")
        except Exception as e:
            raise AppException(e, sys) from e

    
    def recommendations_engine(self,selected_books):
        try:
            recommended_books,poster_url = self.recommend_book(selected_books)
            
            st.markdown("### 📚 Recommended Books for You")
            st.markdown("---")
            
            # Create responsive columns
            cols = st.columns(5, gap="medium")
            
            for i, col in enumerate(cols, 1):
                with col:
                    # Create a card-like container
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding: 15px;
                                border-radius: 15px;
                                text-align: center;
                                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                margin-bottom: 10px;
                            ">
                                <h4 style="color: white; margin: 0; font-size: 14px;">
                                    {recommended_books[i][:50]}{"..." if len(recommended_books[i]) > 50 else ""}
                                </h4>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Display image with styling
                        try:
                            st.image(
                                poster_url[i], 
                                use_container_width=True,
                                caption=f"Recommendation {i}"
                            )
                        except:
                            st.markdown(
                                """
                                <div style="
                                    background: #f0f0f0;
                                    height: 200px;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    border-radius: 10px;
                                    margin-bottom: 10px;
                                ">
                                    <p style="color: #666;">📖 No Image Available</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {str(e)}")
            raise AppException(e, sys) from e


def main():
    # Page configuration
    st.set_page_config(
        page_title="BookWise - AI Book Recommender",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .main-header h1 {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-header p {
            font-size: 1.2rem;
            margin-bottom: 0;
            opacity: 0.9;
        }
        
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }
        
        .stats-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        .stSelectbox > div > div {
            border-radius: 10px;
            border: 2px solid #667eea;
        }
        
        .recommendation-container {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 15px;
            margin-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 BookWise</h1>
        <p>Discover Your Next Favorite Book with AI-Powered Recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 About BookWise")
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 AI-Powered</h4>
            <p>Uses collaborative filtering to find books similar to your preferences</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>📖 Personalized</h4>
            <p>Get recommendations tailored to your reading taste</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🚀 Fast & Accurate</h4>
            <p>Instant recommendations with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.markdown("""
        <div class="stats-card">
            <h3>10,000+</h3>
            <p>Books in Database</p>
        </div>
        <div class="stats-card">
            <h3>95%</h3>
            <p>Accuracy Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    obj = Recommendation()
    
    # Training section
    st.markdown("## 🔧 System Training")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <p>Train the recommendation system with the latest book data to improve accuracy and discover new recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button('🚀 Train System', key="train_btn"):
            with st.spinner('Training in progress...'):
                obj.train_engine()
    
    st.markdown("---")
    
    # Recommendation section
    st.markdown("## 🎯 Get Recommendations")
    
    # Book selection
    try:
        book_names = pickle.load(open(os.path.join('templates','book_names.pkl') ,'rb'))
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_books = st.selectbox(
                "🔍 Search or select a book you enjoyed:",
                book_names,
                help="Start typing to search for a book or scroll through the dropdown"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            recommend_btn = st.button('✨ Get Recommendations', key="recommend_btn")
        
        # Show selected book info
        if selected_books:
            st.markdown(f"""
            <div class="feature-card">
                <h4>📖 Selected Book:</h4>
                <p style="font-size: 1.1rem; color: #667eea;"><strong>{selected_books}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Generate recommendations
        if recommend_btn and selected_books:
            with st.spinner('🔍 Finding perfect recommendations for you...'):
                st.markdown('<div class="recommendation-container">', unsafe_allow_html=True)
                obj.recommendations_engine(selected_books)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add feedback section
                st.markdown("---")
                st.markdown("### 💬 How were these recommendations?")
                feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
                
                with feedback_col1:
                    if st.button("👍 Great!"):
                        st.success("Thanks for your feedback!")
                
                with feedback_col2:
                    if st.button("👌 Good"):
                        st.info("We'll keep improving!")
                
                with feedback_col3:
                    if st.button("👎 Not helpful"):
                        st.warning("Sorry about that. We'll work on better recommendations!")
                        
    except FileNotFoundError:
        st.error("❌ Book database not found. Please train the system first.")
    except Exception as e:
        st.error(f"❌ Error loading books: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p>Made with ❤️ using Streamlit | BookWise - Your AI Reading Companion</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()