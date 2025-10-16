import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import base64
import zipfile
from datetime import datetime
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import io
from PIL import Image, ImageDraw, ImageFont
import random
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Smart Civic Complaint Analyzer",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ComplaintAnalyzer:
    def __init__(self):
        self.excel_file = "complaints.xlsx"
        self.csv_file = "complaints.csv"
        self.photos_dir = "complaint_photos"
        self.model_file = "complaint_classifier.pkl"
        self.vectorizer_file = "tfidf_vectorizer.pkl"
        self.label_encoder_file = "label_encoder.pkl"
        
        # Create photos directory if it doesn't exist
        os.makedirs(self.photos_dir, exist_ok=True)
        
        # Initialize or load models
        self.initialize_models()
        
        # Emergency keywords for urgency detection
        self.emergency_keywords = [
            'emergency', 'urgent', 'immediate', 'critical', 'danger', 'hazard',
            'burst', 'flood', 'fire', 'accident', 'collapse', 'broken',
            'leak', 'spark', 'shock', 'fallen', 'blocked', 'overflow',
            'contamination', 'outage', 'blackout', 'live wire', 'electrocution'
        ]
        
        # High urgency keywords
        self.high_urgency_keywords = [
            'burst pipe', 'water flooding', 'electrical hazard', 'gas leak',
            'road accident', 'building collapse', 'sewage overflow',
            'power outage', 'traffic jam', 'medical emergency'
        ]
        
        # Medium urgency keywords
        self.medium_urgency_keywords = [
            'slow', 'delay', 'complaint', 'request', 'maintenance',
            'repair needed', 'cleaning required', 'garbage accumulation'
        ]

    def initialize_models(self):
        """Initialize or load ML models for classification"""
        # Sample training data
        training_data = [
            ("water supply is not working", "Water"),
            ("no water in my area", "Water"),
            ("water pipe leakage", "Water"),
            ("dirty water coming from tap", "Water"),
            ("low water pressure", "Water"),
            ("power cut since morning", "Electricity"),
            ("electricity outage in sector", "Electricity"),
            ("street light not working", "Electricity"),
            ("electrical wiring problem", "Electricity"),
            ("voltage fluctuation issue", "Electricity"),
            ("road full of potholes", "Roads"),
            ("broken road needs repair", "Roads"),
            ("traffic signal not working", "Roads"),
            ("speed breaker required", "Roads"),
            ("road construction debris", "Roads"),
            ("garbage not collected", "Sanitation"),
            ("drainage blockage", "Sanitation"),
            ("sewage problem in locality", "Sanitation"),
            ("public toilet cleaning", "Sanitation"),
            ("mosquito breeding area", "Sanitation"),
            ("noise pollution", "Others"),
            ("park maintenance", "Others"),
            ("stray animal problem", "Others")
        ]
        
        if os.path.exists(self.model_file) and os.path.exists(self.vectorizer_file):
            # Load existing models
            self.vectorizer = joblib.load(self.vectorizer_file)
            self.classifier = joblib.load(self.model_file)
            self.label_encoder = joblib.load(self.label_encoder_file)
        else:
            # Train new models
            texts, labels = zip(*training_data)
            
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            X = self.vectorizer.fit_transform(texts)
            
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(labels)
            
            self.classifier = MultinomialNB()
            self.classifier.fit(X, y)
            
            # Save models
            joblib.dump(self.vectorizer, self.vectorizer_file)
            joblib.dump(self.classifier, self.model_file)
            joblib.dump(self.label_encoder, self.label_encoder_file)

    def preprocess_text(self, text):
        """Preprocess complaint text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def classify_complaint(self, text):
        """Classify complaint into categories"""
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return "Others"
        
        try:
            X = self.vectorizer.transform([processed_text])
            prediction = self.classifier.predict(X)
            category = self.label_encoder.inverse_transform(prediction)[0]
            return category
        except:
            return "Others"

    def detect_urgency(self, text):
        """Detect urgency level based on keywords"""
        text_lower = text.lower()
        
        # Check for emergency keywords
        emergency_count = sum(1 for keyword in self.emergency_keywords if keyword in text_lower)
        high_count = sum(1 for keyword in self.high_urgency_keywords if keyword in text_lower)
        medium_count = sum(1 for keyword in self.medium_urgency_keywords if keyword in text_lower)
        
        if emergency_count >= 2 or high_count >= 1:
            return "High"
        elif medium_count >= 2 or emergency_count == 1:
            return "Medium"
        else:
            return "Low"

    def extract_location_keywords(self, text):
        """Extract potential location keywords using simple pattern matching"""
        # Common location indicators
        location_indicators = [
            'street', 'road', 'lane', 'avenue', 'sector', 'block',
            'area', 'colony', 'nagar', 'village', 'town', 'city',
            'near', 'opposite', 'behind', 'beside', 'at', 'in'
        ]
        
        words = text.lower().split()
        locations = []
        
        for i, word in enumerate(words):
            if word in location_indicators and i + 1 < len(words):
                locations.append(f"{word} {words[i+1]}")
            elif word.replace(',', '').replace('.', '') in location_indicators and i + 1 < len(words):
                locations.append(f"{word} {words[i+1]}")
        
        # Also extract capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        locations.extend(proper_nouns[:2])  # Take first 2 proper nouns as potential locations
        
        return ', '.join(set(locations[:3]))  # Return unique locations, max 3

    def save_uploaded_photo(self, uploaded_file, complaint_id):
        """Save uploaded photo and return filename"""
        if uploaded_file is not None:
            # Generate unique filename
            file_extension = uploaded_file.name.split('.')[-1].lower()
            filename = f"complaint_{complaint_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
            file_path = os.path.join(self.photos_dir, filename)
            
            # Save the file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            return filename
        return None

    def get_photo_preview(self, photo_filename, width=300):
        """Get photo preview for display with proper error handling"""
        try:
            # Handle NaN values and ensure it's a string
            if pd.isna(photo_filename) or not isinstance(photo_filename, str):
                return None
            
            # Check if it's "No Photo" or empty
            if photo_filename in ["No Photo", "", "nan", "None"]:
                return None
            
            file_path = os.path.join(self.photos_dir, photo_filename)
            
            if os.path.exists(file_path):
                image = Image.open(file_path)
                # Resize for preview while maintaining aspect ratio
                image.thumbnail((width, width))
                return image
            else:
                return None
        except Exception as e:
            return None

    def get_all_photos(self):
        """Get all photos from the photos directory"""
        try:
            photos = []
            for filename in os.listdir(self.photos_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    photos.append(filename)
            return sorted(photos)
        except:
            return []

    def create_photos_zip(self):
        """Create a zip file of all complaint photos"""
        try:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for filename in self.get_all_photos():
                    file_path = os.path.join(self.photos_dir, filename)
                    if os.path.exists(file_path):
                        zip_file.write(file_path, filename)
            
            zip_buffer.seek(0)
            return zip_buffer.getvalue()
        except Exception as e:
            return None

    def initialize_data_files(self):
        """Initialize data files if they don't exist"""
        columns = ['Complaint_ID', 'Timestamp', 'Username', 'Phone_Number', 'Complaint_Text', 
                  'Category', 'Urgency', 'Location_Keywords', 'Photo_Filename']
        
        # Initialize Excel file
        if not os.path.exists(self.excel_file):
            df = pd.DataFrame(columns=columns)
            df.to_excel(self.excel_file, index=False)
        
        # Initialize CSV file for better persistence
        if not os.path.exists(self.csv_file):
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_file, index=False)

    def generate_complaint_id(self):
        """Generate unique complaint ID"""
        return f"COMP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

    def save_complaint(self, username, phone_number, complaint_text, category, urgency, location, photo_filename):
        """Save complaint to both CSV and Excel files for persistence"""
        self.initialize_data_files()
        
        # Try to load existing data from CSV first
        try:
            df = pd.read_csv(self.csv_file)
            # Ensure Timestamp is datetime and handle missing columns
            if not df.empty and 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Handle missing columns in old data
            if 'Phone_Number' not in df.columns:
                df['Phone_Number'] = "Not Provided"
            if 'Photo_Filename' not in df.columns:
                df['Photo_Filename'] = "No Photo"
                
        except:
            # If CSV fails, try Excel
            try:
                df = pd.read_excel(self.excel_file)
                # Handle missing columns in old data
                if 'Phone_Number' not in df.columns:
                    df['Phone_Number'] = "Not Provided"
                if 'Photo_Filename' not in df.columns:
                    df['Photo_Filename'] = "No Photo"
            except:
                df = pd.DataFrame(columns=[
                    'Complaint_ID', 'Timestamp', 'Username', 'Phone_Number', 'Complaint_Text', 
                    'Category', 'Urgency', 'Location_Keywords', 'Photo_Filename'
                ])
        
        # Generate complaint ID
        complaint_id = self.generate_complaint_id()
        
        # Create new row
        new_row = {
            'Complaint_ID': complaint_id,
            'Timestamp': datetime.now(),
            'Username': username,
            'Phone_Number': phone_number,
            'Complaint_Text': complaint_text,
            'Category': category,
            'Urgency': urgency,
            'Location_Keywords': location,
            'Photo_Filename': photo_filename if photo_filename else "No Photo"
        }
        
        # Append new row
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to both CSV and Excel for redundancy
        df.to_csv(self.csv_file, index=False)
        df.to_excel(self.excel_file, index=False)
        
        return df, complaint_id

    def load_complaints(self):
        """Load all complaints from CSV file with fallback to Excel"""
        try:
            # Try CSV first for better persistence
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                # Convert Timestamp string to datetime
                if not df.empty and 'Timestamp' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                
                # Handle missing columns in old data
                if 'Phone_Number' not in df.columns:
                    df['Phone_Number'] = "Not Provided"
                if 'Photo_Filename' not in df.columns:
                    df['Photo_Filename'] = "No Photo"
                else:
                    # Clean up any NaN values in Photo_Filename
                    df['Photo_Filename'] = df['Photo_Filename'].fillna("No Photo")
                    
                return df
            # Fallback to Excel
            elif os.path.exists(self.excel_file):
                df = pd.read_excel(self.excel_file)
                # Handle missing columns in old data
                if 'Phone_Number' not in df.columns:
                    df['Phone_Number'] = "Not Provided"
                if 'Photo_Filename' not in df.columns:
                    df['Photo_Filename'] = "No Photo"
                else:
                    # Clean up any NaN values in Photo_Filename
                    df['Photo_Filename'] = df['Photo_Filename'].fillna("No Photo")
                    
                # Save to CSV for future use
                df.to_csv(self.csv_file, index=False)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading complaints: {e}")
            return pd.DataFrame()

    def get_sentiment(self, text):
        """Get sentiment analysis using TextBlob"""
        try:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            
            if polarity > 0.1:
                return "Positive"
            elif polarity < -0.1:
                return "Negative"
            else:
                return "Neutral"
        except:
            return "Neutral"

    def generate_wordcloud(self, df):
        """Generate word cloud from all complaints with stopwords removed"""
        if df.empty:
            # Create a simple placeholder image
            img = Image.new('RGB', (800, 400), color='white')
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
                draw.text((400, 200), "No complaints yet", fill="black", font=font, anchor="mm")
            except:
                draw.text((400, 200), "No complaints yet", fill="black", anchor="mm")
            return img
        
        # Combine all complaint texts
        text = ' '.join(df['Complaint_Text'].astype(str))
        
        if not text.strip():
            img = Image.new('RGB', (800, 400), color='white')
            return img
        
        try:
            # Preprocess text
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Comprehensive stop words list
            stop_words = {
                'the', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'with', 'on', 'at', 'by', 
                'this', 'that', 'are', 'as', 'be', 'was', 'were', 'has', 'have', 'had', 'but', 
                'not', 'we', 'they', 'you', 'i', 'he', 'she', 'his', 'her', 'our', 'my', 'your', 
                'their', 'its', 'from', 'or', 'an', 'a', 'if', 'will', 'would', 'could', 'should',
                'when', 'where', 'how', 'what', 'why', 'which', 'who', 'whom', 'there', 'here',
                'been', 'have', 'has', 'had', 'do', 'does', 'did', 'done', 'doing', 'can', 'could',
                'may', 'might', 'must', 'shall', 'should', 'will', 'would', 'get', 'got', 'getting',
                'like', 'just', 'more', 'so', 'than', 'then', 'them', 'these', 'those', 'too',
                'very', 'also', 'any', 'both', 'each', 'few', 'most', 'other', 'some', 'such',
                'no', 'nor', 'own', 'same', 'see', 'now', 'one', 'two', 'three', 'first', 'second',
                'new', 'old', 'good', 'bad', 'big', 'small', 'high', 'low', 'long', 'short',
                'great', 'little', 'much', 'many', 'another', 'every', 'all', 'only', 'very'
            }
            
            filtered_words = [word for word in words if word not in stop_words]
            
            # Count word frequencies
            word_freq = Counter(filtered_words)
            top_words = word_freq.most_common(50)
            
            if not top_words:
                img = Image.new('RGB', (800, 400), color='white')
                return img
            
            # Create image
            img_width, img_height = 800, 400
            img = Image.new('RGB', (img_width, img_height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Colors for different word sizes
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # Calculate font sizes based on frequency
            max_freq = max(freq for _, freq in top_words)
            min_freq = min(freq for _, freq in top_words)
            
            positions_used = []
            
            for word, freq in top_words[:30]:
                # Calculate font size based on frequency (scaled)
                if max_freq == min_freq:
                    font_size = 24
                else:
                    font_size = int(16 + (freq - min_freq) / (max_freq - min_freq) * 32)
                
                # Try to use a font, fall back to default
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    try:
                        font = ImageFont.load_default()
                    except:
                        font = None
                
                # Get text size
                if font:
                    bbox = draw.textbbox((0, 0), word, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    text_width = len(word) * font_size * 0.6
                    text_height = font_size
                
                # Try to find a position that doesn't overlap
                max_attempts = 100
                for attempt in range(max_attempts):
                    x = random.randint(0, max(1, img_width - int(text_width)))
                    y = random.randint(0, max(1, img_height - int(text_height)))
                    
                    # Check for overlap
                    overlap = False
                    for (px, py, pwidth, pheight) in positions_used:
                        if (x < px + pwidth and x + text_width > px and
                            y < py + pheight and y + text_height > py):
                            overlap = True
                            break
                    
                    if not overlap:
                        positions_used.append((x, y, text_width, text_height))
                        break
                else:
                    continue
                
                # Choose color
                color = random.choice(colors)
                
                # Draw the text
                if font:
                    draw.text((x, y), word, fill=color, font=font)
                else:
                    draw.text((x, y), word, fill=color)
            
            return img
            
        except Exception as e:
            st.warning(f"Word cloud generation failed: {e}")
            img = Image.new('RGB', (800, 400), color='white')
            return img

def main():
    st.title("🏙️ Smart Civic Complaint Analyzer")
    st.markdown("""
    A comprehensive platform for citizens to report civic issues with AI-powered analysis 
    for faster municipal response and community transparency.
    """)
    
    # Initialize analyzer
    analyzer = ComplaintAnalyzer()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a section",
        ["Submit Complaint", "View All Complaints", "Analytics Dashboard", "Photo Gallery", "AI Insights", "Download Data"]
    )
    
    # Load existing complaints
    complaints_df = analyzer.load_complaints()
    
    if app_mode == "Submit Complaint":
        render_complaint_submission(analyzer, complaints_df)
    
    elif app_mode == "View All Complaints":
        render_complaint_view(analyzer, complaints_df)
    
    elif app_mode == "Analytics Dashboard":
        render_analytics_dashboard(analyzer, complaints_df)
    
    elif app_mode == "Photo Gallery":
        render_photo_gallery(analyzer, complaints_df)
    
    elif app_mode == "AI Insights":
        render_ai_insights(analyzer, complaints_df)
    
    elif app_mode == "Download Data":
        render_download_section(analyzer, complaints_df)

def render_complaint_submission(analyzer, complaints_df):
    """Render complaint submission section"""
    st.header("📝 Submit a Complaint")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        username = st.text_input("Your Name (Optional)", placeholder="Enter your name")
        
        # Phone number field
        phone_number = st.text_input("Phone Number (Optional)", placeholder="Enter your phone number")
        
        complaint_text = st.text_area(
            "Describe your complaint in detail",
            placeholder="Example: There's a major water pipe burst near Main Street causing flooding in the area. This is an emergency situation!",
            height=150
        )
        
        # Photo upload section
        st.subheader("📷 Upload Complaint Photo (Optional)")
        uploaded_photo = st.file_uploader(
            "Add visual evidence of the issue", 
            type=['jpg', 'jpeg', 'png', 'gif'],
            help="Upload a photo showing the problem for better understanding"
        )
        
        if uploaded_photo is not None:
            st.image(uploaded_photo, caption="📸 Photo Preview", width=300)
    
    with col2:
        st.subheader("💡 Tips for Effective Complaints")
        st.markdown("""
        - Be specific about location
        - Mention urgency clearly  
        - Describe the issue in detail
        - Include relevant landmarks
        - **Add photos for better context**
        """)
        
        st.subheader("🚨 Emergency Keywords")
        st.markdown("""
        Use these words for urgent issues:
        - Emergency, Urgent, Critical
        - Burst, Flood, Fire
        - Hazard, Danger, Accident
        """)
        
        st.subheader("📞 Contact Information")
        st.markdown("""
        Providing phone number helps:
        - Faster resolution updates
        - Verification if needed
        - Emergency contact
        - Your data is secure with us
        """)
    
    # Process single complaint
    if st.button("Submit Complaint", type="primary"):
        if complaint_text.strip():
            with st.spinner("Analyzing your complaint..."):
                # Analyze complaint
                category = analyzer.classify_complaint(complaint_text)
                urgency = analyzer.detect_urgency(complaint_text)
                location = analyzer.extract_location_keywords(complaint_text)
                
                # Save photo if uploaded
                photo_filename = None
                if uploaded_photo is not None:
                    temp_id = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    photo_filename = analyzer.save_uploaded_photo(uploaded_photo, temp_id)
                
                # Save complaint
                updated_df, complaint_id = analyzer.save_complaint(
                    username if username else "Anonymous",
                    phone_number if phone_number else "Not Provided",
                    complaint_text,
                    category,
                    urgency,
                    location,
                    photo_filename
                )
                
                # If photo was saved with temp ID, rename it with actual complaint ID
                if photo_filename and "temp_" in photo_filename:
                    new_filename = photo_filename.replace("temp_", f"{complaint_id}_")
                    old_path = os.path.join(analyzer.photos_dir, photo_filename)
                    new_path = os.path.join(analyzer.photos_dir, new_filename)
                    if os.path.exists(old_path):
                        os.rename(old_path, new_path)
                    
                    # Update the dataframe with correct filename
                    updated_df.loc[updated_df['Complaint_ID'] == complaint_id, 'Photo_Filename'] = new_filename
                    updated_df.to_csv(analyzer.csv_file, index=False)
                    updated_df.to_excel(analyzer.excel_file, index=False)
                    photo_filename = new_filename
                
                # Display results
                st.success("✅ Complaint submitted successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.info(f"**Complaint ID:** {complaint_id}")
                with col2:
                    st.info(f"**Category:** {category}")
                with col3:
                    urgency_color = "🔴" if urgency == "High" else "🟡" if urgency == "Medium" else "🟢"
                    st.info(f"**Urgency:** {urgency_color} {urgency}")
                with col4:
                    st.info(f"**Location:** {location if location else 'Not specified'}")
                
                # Show contact info summary
                if phone_number:
                    st.info(f"**📞 Contact:** {phone_number}")
                
                # Show photo if uploaded
                if photo_filename and photo_filename != "No Photo":
                    st.subheader("📷 Uploaded Photo")
                    photo_preview = analyzer.get_photo_preview(photo_filename)
                    if photo_preview:
                        st.image(photo_preview, caption="Your uploaded photo", use_column_width=True)
        else:
            st.error("Please enter a complaint description")

def render_photo_gallery(analyzer, complaints_df):
    """Render photo gallery section"""
    st.header("🖼️ Complaint Photo Gallery")
    
    # Get all photos
    all_photos = analyzer.get_all_photos()
    
    if not all_photos:
        st.info("No photos uploaded yet. Submit a complaint with photos to see them here!")
        return
    
    st.subheader(f"📸 Found {len(all_photos)} complaint photos")
    
    # Filter complaints with photos
    complaints_with_photos = complaints_df[
        (complaints_df['Photo_Filename'] != "No Photo") & 
        (complaints_df['Photo_Filename'].notna())
    ]
    
    if complaints_with_photos.empty:
        st.info("No complaint photos found in the database.")
        return
    
    # Display photos in a grid
    cols = st.columns(3)
    for idx, (_, row) in enumerate(complaints_with_photos.iterrows()):
        photo_filename = row['Photo_Filename']
        if pd.notna(photo_filename) and isinstance(photo_filename, str):
            photo_preview = analyzer.get_photo_preview(photo_filename, width=250)
            if photo_preview:
                with cols[idx % 3]:
                    st.image(photo_preview, use_column_width=True)
                    st.caption(f"**{row['Complaint_ID']}**")
                    st.caption(f"Category: {row['Category']} | Urgency: {row['Urgency']}")
                    st.caption(f"Date: {row['Timestamp'].strftime('%Y-%m-%d')}")

def render_complaint_view(analyzer, complaints_df):
    """Render all complaints view with photos"""
    st.header("📋 All Complaints")
    
    if complaints_df.empty:
        st.info("No complaints submitted yet. Be the first to submit a complaint!")
        return
    
    # Fix any NaN values in Complaint_ID before displaying
    complaints_df['Complaint_ID'] = complaints_df['Complaint_ID'].fillna('Unknown_ID')
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        category_filter = st.multiselect(
            "Filter by Category",
            options=complaints_df['Category'].unique(),
            default=complaints_df['Category'].unique()
        )
    with col2:
        urgency_filter = st.multiselect(
            "Filter by Urgency",
            options=complaints_df['Urgency'].unique(),
            default=complaints_df['Urgency'].unique()
        )
    with col3:
        search_text = st.text_input("Search in complaints")
    with col4:
        sort_by = st.selectbox(
            "Sort by",
            ["Timestamp (Newest)", "Timestamp (Oldest)", "Urgency", "Category"]
        )
    
    # Apply filters
    filtered_df = complaints_df.copy()
    if category_filter:
        filtered_df = filtered_df[filtered_df['Category'].isin(category_filter)]
    if urgency_filter:
        filtered_df = filtered_df[filtered_df['Urgency'].isin(urgency_filter)]
    if search_text:
        filtered_df = filtered_df[
            filtered_df['Complaint_Text'].str.contains(search_text, case=False, na=False) |
            filtered_df['Location_Keywords'].str.contains(search_text, case=False, na=False) |
            filtered_df['Complaint_ID'].str.contains(search_text, case=False, na=False)
        ]
    
    # Apply sorting
    if sort_by == "Timestamp (Newest)":
        filtered_df = filtered_df.sort_values('Timestamp', ascending=False)
    elif sort_by == "Timestamp (Oldest)":
        filtered_df = filtered_df.sort_values('Timestamp', ascending=True)
    elif sort_by == "Urgency":
        urgency_order = {'High': 3, 'Medium': 2, 'Low': 1}
        filtered_df['Urgency_Order'] = filtered_df['Urgency'].map(urgency_order)
        filtered_df = filtered_df.sort_values('Urgency_Order', ascending=False)
        filtered_df = filtered_df.drop('Urgency_Order', axis=1)
    elif sort_by == "Category":
        filtered_df = filtered_df.sort_values('Category')
    
    # Display statistics
    st.subheader(f"📊 Showing {len(filtered_df)} complaints")
    
    # Display each complaint with photo
    for idx, row in filtered_df.iterrows():
        complaint_id = str(row['Complaint_ID']) if pd.notna(row['Complaint_ID']) else "Unknown_ID"
        category = str(row['Category']) if pd.notna(row['Category']) else "Unknown"
        urgency = str(row['Urgency']) if pd.notna(row['Urgency']) else "Unknown"
        
        with st.expander(f"🔸 {complaint_id} - {category} - {urgency} Urgency", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**📝 Complaint:** {row['Complaint_Text']}")
                st.write(f"**👤 Submitted by:** {row['Username']}")
                if pd.notna(row['Phone_Number']) and row['Phone_Number'] != "Not Provided":
                    st.write(f"**📞 Phone:** {row['Phone_Number']}")
                st.write(f"**📍 Location:** {row['Location_Keywords']}")
                st.write(f"**🕒 Submitted on:** {row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Urgency badge
                urgency_color = "🔴" if row['Urgency'] == "High" else "🟡" if row['Urgency'] == "Medium" else "🟢"
                st.write(f"**🚨 Urgency Level:** {urgency_color} {row['Urgency']}")
                
                # Category badge
                st.write(f"**📊 Category:** {row['Category']}")
            
            with col2:
                # Display photo if available
                photo_filename = row['Photo_Filename']
                if pd.notna(photo_filename) and isinstance(photo_filename, str) and photo_filename not in ["No Photo", "", "nan"]:
                    photo_preview = analyzer.get_photo_preview(photo_filename)
                    if photo_preview:
                        st.image(photo_preview, caption="📷 Complaint Photo", use_column_width=True)
                        st.info(f"**Photo:** {photo_filename}")
                    else:
                        st.info("📷 Photo file not found")
                else:
                    st.info("📷 No photo uploaded")

def render_analytics_dashboard(analyzer, complaints_df):
    """Render enhanced analytics dashboard with NLP visualizations"""
    st.header("📈 Enhanced Analytics Dashboard")
    
    if complaints_df.empty:
        st.info("No data available for analytics. Submit some complaints first!")
        return
    
    # Add sentiment analysis to complaints data
    complaints_df['Sentiment'] = complaints_df['Complaint_Text'].apply(analyzer.get_sentiment)
    
    # Key metrics
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_complaints = len(complaints_df)
        st.metric("Total Complaints", total_complaints)
    
    with col2:
        high_urgency = len(complaints_df[complaints_df['Urgency'] == 'High'])
        st.metric("High Urgency Complaints", high_urgency)
    
    with col3:
        unique_users = complaints_df['Username'].nunique()
        st.metric("Unique Users", unique_users)
    
    with col4:
        with_photos = len(complaints_df[
            (complaints_df['Photo_Filename'] != "No Photo") & 
            (complaints_df['Photo_Filename'].notna())
        ])
        st.metric("Complaints with Photos", with_photos)
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment Distribution
        st.subheader("😊 Sentiment Analysis")
        sentiment_counts = complaints_df['Sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Urgency distribution
        st.subheader("🚨 Complaints by Urgency")
        urgency_counts = complaints_df['Urgency'].value_counts()
        fig = px.bar(
            x=urgency_counts.index,
            y=urgency_counts.values,
            color=urgency_counts.index,
            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
            labels={'x': 'Urgency Level', 'y': 'Number of Complaints'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category distribution
        st.subheader("📊 Complaints by Category")
        category_counts = complaints_df['Category'].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent complaints trend
        st.subheader("📈 Complaints Over Time")
        complaints_df['Date'] = pd.to_datetime(complaints_df['Timestamp']).dt.date
        daily_complaints = complaints_df.groupby('Date').size().reset_index(name='Count')
        fig = px.area(
            daily_complaints,
            x='Date',
            y='Count',
            title='Daily Complaint Trends'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Category vs Urgency Heatmap
        st.subheader("🔥 Category vs Urgency Heatmap")
        heatmap_data = pd.crosstab(complaints_df['Category'], complaints_df['Urgency'])
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Urgency Level", y="Category", color="Count"),
            aspect="auto",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top Keywords Frequency
        st.subheader("🔤 Top Complaint Keywords")
        all_text = ' '.join(complaints_df['Complaint_Text'].astype(str))
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        stop_words = {'this', 'that', 'with', 'have', 'from', 'they', 'when', 'were', 'been', 'also'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        word_freq = Counter(filtered_words).most_common(10)
        
        if word_freq:
            keywords, counts = zip(*word_freq)
            fig = px.bar(
                x=counts,
                y=keywords,
                orientation='h',
                title='Most Frequent Keywords',
                labels={'x': 'Frequency', 'y': 'Keywords'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Third row - Word Cloud
    st.subheader("☁️ Word Cloud - Most Frequent Issues")
    wordcloud_img = analyzer.generate_wordcloud(complaints_df)
    st.image(wordcloud_img, use_column_width=True)

def render_ai_insights(analyzer, complaints_df):
    """Render AI Insights section with NLP-focused visualizations"""
    st.header("🤖 AI Insights & NLP Analysis")
    
    if complaints_df.empty:
        st.info("No complaints data available. Submit some complaints to see AI insights.")
        return
    
    # Add sentiment analysis
    complaints_df['Sentiment'] = complaints_df['Complaint_Text'].apply(analyzer.get_sentiment)
    
    st.subheader("📊 Text Analysis Visualizations")
    
    # First row
    col1, col2 = st.columns(2)
    
    with col1:
        # Text Length Distribution
        st.subheader("📏 Complaint Text Length Distribution")
        complaints_df['Text_Length'] = complaints_df['Complaint_Text'].str.len()
        fig = px.histogram(
            complaints_df,
            x='Text_Length',
            nbins=20,
            title='Distribution of Complaint Text Lengths',
            labels={'Text_Length': 'Number of Characters'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment by Category
        st.subheader("😊 Sentiment by Category")
        sentiment_by_category = pd.crosstab(complaints_df['Category'], complaints_df['Sentiment'])
        fig = px.bar(
            sentiment_by_category,
            barmode='group',
            title='Sentiment Distribution Across Categories',
            color_discrete_map={'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Word Frequency by Urgency
        st.subheader("🔤 Top Words by Urgency Level")
        
        # Get top words for each urgency level
        urgency_levels = complaints_df['Urgency'].unique()
        
        for urgency in urgency_levels:
            urgency_text = ' '.join(complaints_df[complaints_df['Urgency'] == urgency]['Complaint_Text'].astype(str))
            words = re.findall(r'\b[a-zA-Z]{4,}\b', urgency_text.lower())
            stop_words = {'this', 'that', 'with', 'have', 'from', 'they', 'when', 'were', 'been', 'also', 'just', 'like'}
            filtered_words = [word for word in words if word not in stop_words]
            word_freq = Counter(filtered_words).most_common(5)
            
            if word_freq:
                words, counts = zip(*word_freq)
                fig = px.bar(
                    x=counts,
                    y=words,
                    orientation='h',
                    title=f'Top Words - {urgency} Urgency',
                    color=counts,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Second row
    col1, col2 = st.columns(2)
    
    with col1:
        # Response Time Simulation (based on urgency)
        st.subheader("⏱️ Estimated Response Time by Urgency")
        
        # Simulate response times based on urgency
        response_times = {
            'High': '1-4 hours',
            'Medium': '24-48 hours', 
            'Low': '3-7 days'
        }
        
        urgency_response = complaints_df['Urgency'].value_counts()
        response_data = []
        for urgency, count in urgency_response.items():
            response_data.append({
                'Urgency': urgency,
                'Count': count,
                'Estimated_Response': response_times.get(urgency, 'Unknown')
            })
        
        response_df = pd.DataFrame(response_data)
        fig = px.bar(
            response_df,
            x='Urgency',
            y='Count',
            color='Estimated_Response',
            title='Complaints by Urgency & Estimated Response Time',
            labels={'Count': 'Number of Complaints'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category Trends Over Time
        st.subheader("📈 Category Trends Over Time")
        
        if len(complaints_df) > 1:
            complaints_df['Week'] = pd.to_datetime(complaints_df['Timestamp']).dt.to_period('W').astype(str)
            weekly_trends = complaints_df.groupby(['Week', 'Category']).size().reset_index(name='Count')
            
            fig = px.line(
                weekly_trends,
                x='Week',
                y='Count',
                color='Category',
                title='Weekly Complaint Trends by Category',
                markers=True
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Third row - Enhanced Word Cloud
    st.subheader("☁️ Enhanced Word Cloud - Key Issues Identified")
    st.info("This word cloud shows the most frequent complaint keywords after removing common stop words")
    
    wordcloud_img = analyzer.generate_wordcloud(complaints_df)
    st.image(wordcloud_img, use_column_width=True)
    
    # Additional NLP metrics
    st.subheader("📝 Text Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_length = complaints_df['Complaint_Text'].str.len().mean()
        st.metric("Average Complaint Length", f"{avg_length:.0f} characters")
    
    with col2:
        total_words = sum(len(str(text).split()) for text in complaints_df['Complaint_Text'])
        st.metric("Total Words Analyzed", f"{total_words:,}")
    
    with col3:
        unique_words = len(set(' '.join(complaints_df['Complaint_Text'].astype(str)).lower().split()))
        st.metric("Unique Words Used", f"{unique_words:,}")

def render_download_section(analyzer, complaints_df):
    """Render download section"""
    st.header("📥 Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Download Complaints Data")
        if not complaints_df.empty:
            # Convert DataFrame to CSV for download
            csv_data = complaints_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV File",
                data=csv_data,
                file_name="complaints_data.csv",
                mime="text/csv",
                help="Download all complaints as CSV file (includes photo filenames)"
            )
            
            # Create Excel file in memory for download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                complaints_df.to_excel(writer, index=False, sheet_name='Complaints')
            excel_data = output.getvalue()
            
            st.download_button(
                label="📊 Download Excel File (XLSX)",
                data=excel_data,
                file_name="complaints_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download all complaints as Excel file (includes photo filenames)"
            )
            
            # Download photos as zip
            st.subheader("📸 Download Photos")
            photos_exist = any((complaints_df['Photo_Filename'] != "No Photo") & complaints_df['Photo_Filename'].notna())
            if photos_exist:
                zip_data = analyzer.create_photos_zip()
                if zip_data:
                    st.download_button(
                        label="🗂️ Download All Photos (ZIP)",
                        data=zip_data,
                        file_name="complaint_photos.zip",
                        mime="application/zip",
                        help="Download all complaint photos as a ZIP file"
                    )
                else:
                    st.warning("Could not create photos zip file")
            else:
                st.info("No photos available for download")
            
            st.info("""
            **📝 Download Notes:**
            - **CSV/Excel files** contain photo filenames, not actual images
            - **ZIP file** contains all actual complaint photos
            - Match photos to complaints using the filenames in the data files
            """)
            
            st.info(f"Total records available: {len(complaints_df)}")
        else:
            st.info("No complaints data available for download")
    
    with col2:
        st.subheader("Download Visualizations")
        if not complaints_df.empty:
            wordcloud_img = analyzer.generate_wordcloud(complaints_df)
            
            img_buffer = io.BytesIO()
            wordcloud_img.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            st.download_button(
                label="🖼️ Download Word Cloud",
                data=img_data,
                file_name="complaint_wordcloud.png",
                mime="image/png",
                help="Download the current word cloud image"
            )
            st.image(wordcloud_img, use_column_width=True)
        else:
            st.info("Word cloud will be available after first complaint")

if __name__ == "__main__":
    main()