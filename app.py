import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
        self.csv_file = "complaints.csv"  # Added CSV for better persistence
        self.model_file = "complaint_classifier.pkl"
        self.vectorizer_file = "tfidf_vectorizer.pkl"
        self.label_encoder_file = "label_encoder.pkl"
        
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

    def initialize_data_files(self):
        """Initialize data files if they don't exist"""
        columns = ['Timestamp', 'Username', 'Complaint_Text', 'Category', 'Urgency', 'Location_Keywords']
        
        # Initialize Excel file
        if not os.path.exists(self.excel_file):
            df = pd.DataFrame(columns=columns)
            df.to_excel(self.excel_file, index=False)
        
        # Initialize CSV file for better persistence
        if not os.path.exists(self.csv_file):
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_file, index=False)

    def save_complaint(self, username, complaint_text, category, urgency, location):
        """Save complaint to both CSV and Excel files for persistence"""
        self.initialize_data_files()
        
        # Try to load existing data from CSV first
        try:
            df = pd.read_csv(self.csv_file)
            # Ensure Timestamp is datetime
            if not df.empty and 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        except:
            # If CSV fails, try Excel
            try:
                df = pd.read_excel(self.excel_file)
            except:
                df = pd.DataFrame(columns=[
                    'Timestamp', 'Username', 'Complaint_Text', 
                    'Category', 'Urgency', 'Location_Keywords'
                ])
        
        # Create new row
        new_row = {
            'Timestamp': datetime.now(),
            'Username': username,
            'Complaint_Text': complaint_text,
            'Category': category,
            'Urgency': urgency,
            'Location_Keywords': location
        }
        
        # Append new row
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to both CSV and Excel for redundancy
        df.to_csv(self.csv_file, index=False)
        df.to_excel(self.excel_file, index=False)
        
        return df

    def load_complaints(self):
        """Load all complaints from CSV file with fallback to Excel"""
        try:
            # Try CSV first for better persistence
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                # Convert Timestamp string to datetime
                if not df.empty and 'Timestamp' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                return df
            # Fallback to Excel
            elif os.path.exists(self.excel_file):
                df = pd.read_excel(self.excel_file)
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

    def get_classification_metrics(self, complaints_df):
        """Calculate classification metrics for the urgency model"""
        if complaints_df.empty or len(complaints_df) < 3:
            return None, None
        
        try:
            # For demonstration, we'll use the actual urgency as ground truth
            # In a real scenario, you'd have labeled test data
            y_true = complaints_df['Urgency']
            
            # Get predictions using the current model
            y_pred = [self.detect_urgency(text) for text in complaints_df['Complaint_Text']]
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=['Low', 'Medium', 'High'])
            
            return metrics, cm
            
        except Exception as e:
            st.warning(f"Could not calculate metrics: {e}")
            return None, None

    def generate_wordcloud(self, df):
        """Generate word cloud from all complaints using PIL (no wordcloud library)"""
        if df.empty:
            # Create a simple placeholder image
            img = Image.new('RGB', (800, 400), color='white')
            draw = ImageDraw.Draw(img)
            # Try to use a default font, or use None if not available
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
            
            # Remove common stop words
            stop_words = {'the', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'with', 'on', 'at', 'by', 'this', 'that', 'are', 'as', 'be', 'was', 'were', 'has', 'have', 'had', 'but', 'not', 'we', 'they', 'you', 'i', 'he', 'she', 'his', 'her', 'our', 'my', 'your', 'their', 'its'}
            filtered_words = [word for word in words if word not in stop_words]
            
            # Count word frequencies
            word_freq = Counter(filtered_words)
            top_words = word_freq.most_common(50)  # Get top 50 words
            
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
            
            for word, freq in top_words[:30]:  # Use top 30 words for clarity
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
                    # If no position found, skip this word
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
            # Return placeholder image
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
        ["Submit Complaint", "View All Complaints", "Analytics Dashboard", "Model Performance", "Download Data"]
    )
    
    # Load existing complaints
    complaints_df = analyzer.load_complaints()
    
    if app_mode == "Submit Complaint":
        render_complaint_submission(analyzer, complaints_df)
    
    elif app_mode == "View All Complaints":
        render_complaint_view(complaints_df)
    
    elif app_mode == "Analytics Dashboard":
        render_analytics_dashboard(analyzer, complaints_df)
    
    elif app_mode == "Model Performance":
        render_model_performance(analyzer, complaints_df)
    
    elif app_mode == "Download Data":
        render_download_section(analyzer, complaints_df)

def render_complaint_submission(analyzer, complaints_df):
    """Render complaint submission section"""
    st.header("📝 Submit a Complaint")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        username = st.text_input("Your Name (Optional)", placeholder="Enter your name")
        complaint_text = st.text_area(
            "Describe your complaint in detail",
            placeholder="Example: There's a major water pipe burst near Main Street causing flooding in the area. This is an emergency situation!",
            height=150
        )
        
        # Batch upload option
        st.subheader("📁 Batch Upload (Optional)")
        uploaded_file = st.file_uploader("Upload CSV file with multiple complaints", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                if st.button("Process Batch Complaints"):
                    process_batch_complaints(analyzer, batch_df)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
    
    with col2:
        st.subheader("💡 Tips for Effective Complaints")
        st.markdown("""
        - Be specific about location
        - Mention urgency clearly
        - Describe the issue in detail
        - Include relevant landmarks
        """)
        
        st.subheader("🚨 Emergency Keywords")
        st.markdown("""
        Use these words for urgent issues:
        - Emergency, Urgent, Critical
        - Burst, Flood, Fire
        - Hazard, Danger, Accident
        """)
    
    # Process single complaint
    if st.button("Submit Complaint", type="primary"):
        if complaint_text.strip():
            with st.spinner("Analyzing your complaint..."):
                # Analyze complaint
                category = analyzer.classify_complaint(complaint_text)
                urgency = analyzer.detect_urgency(complaint_text)
                location = analyzer.extract_location_keywords(complaint_text)
                
                # Save complaint
                updated_df = analyzer.save_complaint(
                    username if username else "Anonymous",
                    complaint_text,
                    category,
                    urgency,
                    location
                )
                
                # Display results
                st.success("✅ Complaint submitted successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Category:** {category}")
                with col2:
                    urgency_color = "🔴" if urgency == "High" else "🟡" if urgency == "Medium" else "🟢"
                    st.info(f"**Urgency:** {urgency_color} {urgency}")
                with col3:
                    st.info(f"**Location:** {location if location else 'Not specified'}")
        else:
            st.error("Please enter a complaint description")

def process_batch_complaints(analyzer, batch_df):
    """Process batch complaints from CSV"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    required_columns = ['Username', 'Complaint_Text']
    if not all(col in batch_df.columns for col in required_columns):
        st.error("CSV must contain 'Username' and 'Complaint_Text' columns")
        return
    
    successful_count = 0
    total_count = len(batch_df)
    
    for i, row in batch_df.iterrows():
        try:
            complaint_text = str(row['Complaint_Text'])
            username = str(row['Username']) if pd.notna(row['Username']) else "Anonymous"
            
            # Analyze complaint
            category = analyzer.classify_complaint(complaint_text)
            urgency = analyzer.detect_urgency(complaint_text)
            location = analyzer.extract_location_keywords(complaint_text)
            
            # Save complaint
            analyzer.save_complaint(username, complaint_text, category, urgency, location)
            successful_count += 1
            
        except Exception as e:
            st.warning(f"Failed to process row {i+1}: {e}")
        
        # Update progress
        progress = (i + 1) / total_count
        progress_bar.progress(progress)
        status_text.text(f"Processed {i+1}/{total_count} complaints")
    
    st.success(f"✅ Successfully processed {successful_count} out of {total_count} complaints!")

def render_complaint_view(complaints_df):
    """Render all complaints view"""
    st.header("📋 All Complaints")
    
    if complaints_df.empty:
        st.info("No complaints submitted yet. Be the first to submit a complaint!")
        return
    
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
            filtered_df['Location_Keywords'].str.contains(search_text, case=False, na=False)
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
    
    # Display complaints in an interactive table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn("Timestamp", format="DD/MM/YYYY HH:mm"),
            "Username": "User",
            "Complaint_Text": "Complaint",
            "Category": "Category",
            "Urgency": "Urgency",
            "Location_Keywords": "Location"
        }
    )

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
        most_common_category = complaints_df['Category'].mode()[0] if not complaints_df.empty else "N/A"
        st.metric("Most Common Issue", most_common_category)
    
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

def render_model_performance(analyzer, complaints_df):
    """Render model performance metrics section"""
    st.header("🤖 Model Performance Metrics")
    
    if complaints_df.empty:
        st.info("No complaints data available. Submit some complaints to see model performance.")
        return
    
    # Get classification metrics
    metrics, cm = analyzer.get_classification_metrics(complaints_df)
    
    if metrics is None:
        st.info("Need at least 3 complaints to calculate model performance metrics.")
        return
    
    # Display metrics in columns
    st.subheader("📊 Classification Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metrics['Precision']:.2%}")
    with col3:
        st.metric("Recall", f"{metrics['Recall']:.2%}")
    with col4:
        st.metric("F1-Score", f"{metrics['F1-Score']:.2%}")
    
    # Confusion Matrix
    st.subheader("🎯 Confusion Matrix")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low', 'Medium', 'High'], 
                   yticklabels=['Low', 'Medium', 'High'],
                   ax=ax)
        ax.set_xlabel('Predicted Urgency')
        ax.set_ylabel('Actual Urgency')
        ax.set_title('Urgency Classification Confusion Matrix')
        st.pyplot(fig)
    
    with col2:
        st.info("""
        **Confusion Matrix Guide:**
        - **Diagonal:** Correct predictions
        - **Off-diagonal:** Misclassifications
        - Colors show prediction density
        """)
    
    # Sample predictions for qualitative analysis
    st.subheader("🔍 Sample Predictions")
    
    if len(complaints_df) >= 3:
        # Display recent complaints with predictions
        sample_complaints = complaints_df.tail(3).copy()
        
        for idx, row in sample_complaints.iterrows():
            with st.expander(f"Complaint: {row['Complaint_Text'][:80]}..."):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Category:** {row['Category']}")
                with col2:
                    st.write(f"**Actual Urgency:** {row['Urgency']}")
                with col3:
                    predicted_urgency = analyzer.detect_urgency(row['Complaint_Text'])
                    status_icon = "✅" if row['Urgency'] == predicted_urgency else "❌"
                    st.write(f"**Predicted Urgency:** {predicted_urgency} {status_icon}")

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
                help="Download all complaints as CSV file"
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
                help="Download all complaints as Excel file"
            )
            
            st.info(f"Total records available: {len(complaints_df)}")
        else:
            st.info("No complaints data available for download")
    
    with col2:
        st.subheader("Download Word Cloud")
        if not complaints_df.empty:
            wordcloud_img = analyzer.generate_wordcloud(complaints_df)
            
            # Convert PIL Image to bytes for download
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