import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from faker import Faker

# Load environment variables from .env file
load_dotenv()


@st.cache_data
def load_data() -> pd.DataFrame:
    # Load data from CSV or generate synthetic dataset if file doesn't exist
    # Determine the path to the data file relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "data.csv")

    if os.path.exists(data_path):
        # Read existing CSV
        df = pd.read_csv(data_path, parse_dates=["order_date"])
    else:
        # Generate synthetic dataset using Faker
        fake = Faker()
        fake.seed_instance(42)
        np.random.seed(42)
        n_orders = 10000
        
        # Define product categories and realistic product names
        categories_products = {
            "Electronics": [
                "iPhone 15 Pro", "Samsung Galaxy S24", "MacBook Air M3", "Dell XPS 13",
                "Sony WH-1000XM5", "AirPods Pro", "iPad Pro", "Nintendo Switch",
                "PlayStation 5", "Apple Watch Series 9", "Kindle Paperwhite",
                "Samsung 4K TV", "LG OLED TV", "Canon EOS R5", "GoPro Hero 12"
            ],
            "Clothing": [
                "Nike Air Max 270", "Adidas Ultraboost 22", "Levi's 501 Jeans", 
                "Champion Hoodie", "Under Armour T-Shirt", "Calvin Klein Underwear",
                "Tommy Hilfiger Polo", "The North Face Jacket", "Converse Chuck Taylor",
                "Vans Old Skool", "Ray-Ban Aviators", "Gucci Belt", "Rolex Submariner",
                "Patagonia Fleece", "Columbia Rain Jacket"
            ],
            "Home & Garden": [
                "Dyson V15 Vacuum", "KitchenAid Stand Mixer", "Ninja Blender",
                "Instant Pot Duo", "Roomba i7+", "Nest Thermostat", "Ring Doorbell",
                "Philips Hue Lights", "Casper Mattress", "West Elm Sofa",
                "IKEA Billy Bookshelf", "Cuisinart Coffee Maker", "Le Creuset Dutch Oven",
                "All-Clad Cookware Set", "Vitamix Blender"
            ],
            "Sports & Outdoors": [
                "Peloton Bike", "Yeti Cooler", "Patagonia Backpack", "REI Hiking Boots",
                "Wilson Tennis Racket", "Spalding Basketball", "Fitbit Charge 5",
                "Garmin GPS Watch", "Coleman Camping Tent", "North Face Sleeping Bag",
                "Hydro Flask Water Bottle", "Lululemon Yoga Mat", "Bowflex Dumbbells",
                "Trek Mountain Bike", "Osprey Hiking Pack"
            ],
            "Books & Media": [
                "Kindle Oasis", "Harry Potter Box Set", "National Geographic Magazine",
                "Adobe Creative Suite", "Microsoft Office 365", "Spotify Premium",
                "Netflix Subscription", "Apple Music", "Audible Subscription",
                "Nintendo Game", "PlayStation Game", "Steam Gift Card",
                "Barnes & Noble Gift Card", "Coursera Course", "MasterClass Subscription"
            ],
            "Health & Beauty": [
                "Fenty Beauty Foundation", "Charlotte Tilbury Lipstick", "The Ordinary Serum",
                "Drunk Elephant Moisturizer", "Glossier Boy Brow", "Rare Beauty Blush",
                "Clinique Cleanser", "Urban Decay Eyeshadow", "MAC Lipstick",
                "Sephora Makeup Brush Set", "CeraVe Facial Cleanser", "Neutrogena Sunscreen",
                "Olaplex Hair Treatment", "Dyson Hair Dryer", "Foreo Luna Cleaner"
            ],
            "Automotive": [
                "Michelin Tires", "Bosch Car Battery", "Garmin Dash Cam", "WeatherTech Floor Mats",
                "Thule Roof Rack", "K&N Air Filter", "Mobil 1 Motor Oil", "Chemical Guys Car Wash",
                "Covercraft Car Cover", "Yakima Bike Rack", "Pioneer Car Stereo",
                "Optima Car Battery", "Rain-X Windshield Treatment", "Armor All Protectant",
                "AutoZone Gift Card"
            ]
        }
        
        # Price ranges for each category (min, max)
        price_ranges = {
            "Electronics": (50, 2000),
            "Clothing": (15, 500),
            "Home & Garden": (25, 800),
            "Sports & Outdoors": (20, 600),
            "Books & Media": (5, 200),
            "Health & Beauty": (12, 200),
            "Automotive": (30, 1200)
        }
        
        records = []
        for i in range(1, n_orders + 1):
            # Select category and product
            category = fake.random_element(elements=list(categories_products.keys()))
            product = fake.random_element(elements=categories_products[category])
            
            # Generate realistic price based on category
            min_price, max_price = price_ranges[category]
            price = round(fake.random.uniform(min_price, max_price), 2)
            
            # Generate quantity (most orders are 1-2 items)
            quantity = fake.random_element(elements=[1, 1, 1, 2, 2, 3, 4, 5])
            
            # Order date between 2015-2025 (10 years of data)
            order_date = fake.date_time_between(start_date='-10y', end_date='now')
            
            # Calculate revenue
            revenue = round(price * quantity, 2)
            
            records.append([i, category, product, price, quantity, order_date, revenue])
            
        df = pd.DataFrame(
            records,
            columns=["order_id", "category", "product_name", "price", "quantity", "order_date", "revenue"],
        )
    return df


def filter_data(df: pd.DataFrame, start_date: datetime, end_date: datetime, categories: list[str]) -> pd.DataFrame:
    # Filter dataset by date range and categories
    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered["order_date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["order_date"] <= pd.to_datetime(end_date)]
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    return filtered


def display_kpis(df: pd.DataFrame) -> None:
    # Calculate and display key performance indicators
    total_sales = float(df["revenue"].sum())
    total_orders = int(df["order_id"].nunique())
    avg_order_value = total_sales / total_orders if total_orders else 0
    orders_per_day = (
        df.groupby(df["order_date"].dt.date).size().mean() if not df.empty else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${total_sales:,.2f}")
    col2.metric("Total Orders", f"{total_orders}")
    col3.metric("Avg Order Value", f"${avg_order_value:,.2f}")
    col4.metric("Orders per Day", f"{orders_per_day:.1f}")


def display_charts(df: pd.DataFrame) -> None:
    # Display interactive charts: revenue over time, revenue by category, and orders by category
    if df.empty:
        st.info("No data to display for the selected filters.")
        return

    # Line chart: revenue over time
    daily_sales = (
        df.groupby(df["order_date"].dt.date)["revenue"].sum().reset_index(name="revenue")
    )
    fig1 = px.line(daily_sales, x="order_date", y="revenue", title="Revenue Over Time")
    fig1.update_layout(
        xaxis_rangeslider_visible=True,  # Enable range slider for zooming
        xaxis=dict(
            type="date",
            rangeslider=dict(visible=True)
        )
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Bar chart: revenue by category
    category_sales = (
        df.groupby("category")["revenue"].sum().reset_index(name="revenue")
    )
    fig2 = px.bar(category_sales, x="category", y="revenue", title="Revenue by Category")
    fig2.update_xaxes(tickangle=45)  # Rotate labels for better readability
    fig2.update_layout(
        xaxis=dict(fixedrange=True),  # Disable zoom on x-axis
        yaxis=dict(fixedrange=True)   # Disable zoom on y-axis
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Pie chart: revenue distribution by category
    fig3 = px.pie(category_sales, values="revenue", names="category", title="Revenue Distribution by Category")
    fig3.update_traces(textposition='inside', textinfo='percent+label')
    fig3.update_layout(
        showlegend=True,
        dragmode=False  # Disable dragging/zooming
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Additional chart: Orders by category
    category_orders = (
        df.groupby("category")["order_id"].count().reset_index(name="orders")
    )
    fig4 = px.bar(category_orders, x="category", y="orders", title="Number of Orders by Category")
    fig4.update_xaxes(tickangle=45)
    fig4.update_layout(
        xaxis=dict(fixedrange=True),  # Disable zoom on x-axis
        yaxis=dict(fixedrange=True)   # Disable zoom on y-axis
    )
    st.plotly_chart(fig4, use_container_width=True)


def generate_ai_insights(df: pd.DataFrame) -> str:
    # Generate AI summary using OpenAI API
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY not set. Add your OpenAI API key to enable AI insights."
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        return "openai package not installed. Run: pip install openai"

    # Get basic stats for the prompt
    total_sales = float(df["revenue"].sum())
    total_orders = int(df["order_id"].nunique())
    avg_order_value = total_sales / total_orders if total_orders else 0
    top_category = (
        df.groupby("category")["revenue"].sum().idxmax() if not df.empty else "N/A"
    )

    # Build prompt for AI analysis
    prompt = (
        "Write a business analysis report for this e-commerce data. Use simple, clear sentences.\n\n"
        f"Business Performance Summary:\n"
        f"The company generated ${total_sales:,.2f} in total revenue.\n"
        f"This came from {total_orders} customer orders.\n" 
        f"The average order value is ${avg_order_value:,.2f}.\n"
        f"The top-performing category is {top_category}.\n\n"
        "Analyze what these numbers mean for the business and provide actionable recommendations."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Failed to generate insights: {e}"


def main() -> None:
    # Main app entry point
    st.set_page_config(
        page_title="E-commerce Analytics Dashboard",
        layout="wide",
    )
    st.title("E-commerce Analytics Dashboard")

    # Load or generate data
    df = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
    )
    available_categories = df["category"].unique().tolist()
    selected_categories = st.sidebar.multiselect(
        "Select Categories", available_categories, default=available_categories
    )

    # Unpack the date range
    if isinstance(date_range, list) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Filter the data based on user inputs
    filtered_df = filter_data(df, start_date, end_date, selected_categories)

    # Display KPIs and charts
    st.subheader("Key Performance Indicators")
    display_kpis(filtered_df)
    st.subheader("Visualizations")
    display_charts(filtered_df)

    # AI Insights Button
    if st.sidebar.button("Generate AI Insights"):
        with st.spinner():
            insights = generate_ai_insights(filtered_df)
        st.subheader("AI-Generated Insights")
        st.write(insights)


if __name__ == "__main__":
    main()















##Ansh Gopinath