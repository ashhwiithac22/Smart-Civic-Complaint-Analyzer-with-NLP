import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import os
from datetime import datetime
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import joblib
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import warnings
import io
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
        self.wordcloud_file = "complaint_cloud.png"
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

    def initialize_excel_file(self):
        """Initialize Excel file if it doesn't exist"""
        if not os.path.exists(self.excel_file):
            df = pd.DataFrame(columns=[
                'Timestamp', 'Username', 'Complaint_Text', 
                'Category', 'Urgency', 'Location_Keywords'
            ])
            df.to_excel(self.excel_file, index=False)

    def save_complaint(self, username, complaint_text, category, urgency, location):
        """Save complaint to Excel file"""
        self.initialize_excel_file()
        
        # Read existing data
        df = pd.read_excel(self.excel_file)
        
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
        
        # Save back to Excel
        df.to_excel(self.excel_file, index=False)
        
        return df

    def generate_wordcloud(self, df):
        """Generate word cloud from all complaints"""
        if df.empty:
            # Create empty word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, 'No complaints yet', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=16)
            ax.axis('off')
            plt.savefig(self.wordcloud_file, bbox_inches='tight', dpi=300, 
                       facecolor='white', transparent=False)
            plt.close()
            return
        
        # Combine all complaint texts
        text = ' '.join(df['Complaint_Text'].astype(str))
        
        if not text.strip():
            return
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            contour_width=1,
            contour_color='steelblue'
        ).generate(text)
        
        # Plot and save
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Frequent Complaint Topics', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(self.wordcloud_file, bbox_inches='tight', dpi=300, 
                   facecolor='white', transparent=False)
        plt.close()

    def load_complaints(self):
        """Load all complaints from Excel file"""
        if os.path.exists(self.excel_file):
            return pd.read_excel(self.excel_file)
        else:
            return pd.DataFrame()

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
        ["Submit Complaint", "View All Complaints", "Analytics Dashboard", "Download Data"]
    )
    
    # Load existing complaints
    complaints_df = analyzer.load_complaints()
    
    if app_mode == "Submit Complaint":
        render_complaint_submission(analyzer, complaints_df)
    
    elif app_mode == "View All Complaints":
        render_complaint_view(complaints_df)
    
    elif app_mode == "Analytics Dashboard":
        render_analytics_dashboard(analyzer, complaints_df)
    
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
                
                # Generate updated word cloud
                analyzer.generate_wordcloud(updated_df)
                
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
    
    # Generate final word cloud
    updated_df = analyzer.load_complaints()
    analyzer.generate_wordcloud(updated_df)
    
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
    """Render analytics dashboard"""
    st.header("📈 Analytics Dashboard")
    
    if complaints_df.empty:
        st.info("No data available for analytics. Submit some complaints first!")
        return
    
    # Generate word cloud
    analyzer.generate_wordcloud(complaints_df)
    
    # Layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        st.subheader("Complaints by Category")
        category_counts = complaints_df['Category'].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Urgency distribution
        st.subheader("Complaints by Urgency")
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
        # Display word cloud
        st.subheader("Word Cloud - Most Frequent Issues")
        if os.path.exists(analyzer.wordcloud_file):
            st.image(analyzer.wordcloud_file, use_column_width=True)
        else:
            st.info("Word cloud will be generated after first complaint")
        
        # Recent complaints trend
        st.subheader("Recent Complaints Trend")
        complaints_df['Date'] = pd.to_datetime(complaints_df['Timestamp']).dt.date
        daily_complaints = complaints_df.groupby('Date').size().reset_index(name='Count')
        fig = px.line(
            daily_complaints,
            x='Date',
            y='Count',
            title='Daily Complaints Trend'
        )
        st.plotly_chart(fig, use_container_width=True)
    
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
        if os.path.exists(analyzer.wordcloud_file):
            with open(analyzer.wordcloud_file, "rb") as file:
                image_data = file.read()
            
            st.download_button(
                label="🖼️ Download Word Cloud",
                data=image_data,
                file_name="complaint_wordcloud.png",
                mime="image/png",
                help="Download the current word cloud image"
            )
            st.image(analyzer.wordcloud_file, use_column_width=True)
        else:
            st.info("Word cloud will be available after first complaint")

if __name__ == "__main__":
    main()