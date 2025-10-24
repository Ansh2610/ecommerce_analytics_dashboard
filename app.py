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

    # Create uniform columns for all KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"${total_sales:,.2f}")
    
    with col2:
        st.metric("Total Orders", f"{total_orders:,}")
    
    with col3:
        st.metric("Avg Order Value", f"${avg_order_value:,.2f}")
    
    with col4:
        st.metric("Orders per Day", f"{orders_per_day:.1f}")


def display_revenue_chart(df: pd.DataFrame) -> None:
    # Revenue over time line chart
    st.subheader("Revenue Trends")
    
    if df.empty:
        st.info("No data to display for the selected filters.")
        return

    daily_sales = (
        df.groupby(df["order_date"].dt.date)["revenue"].sum().reset_index(name="revenue")
    )
    fig1 = px.line(daily_sales, x="order_date", y="revenue", title="Revenue Over Time")
    
    # Enhanced styling
    fig1.update_traces(
        line=dict(color='#1f77b4', width=3),
        hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:,.2f}<extra></extra>'
    )
    
    fig1.update_layout(
        xaxis_rangeslider_visible=True,
        xaxis=dict(
            type="date",
            rangeslider=dict(visible=True),
            title="Date",
            title_font=dict(size=14, color='#2c3e50'),
            tickfont=dict(size=12, color='#34495e')
        ),
        yaxis=dict(
            title="Revenue ($)",
            title_font=dict(size=14, color='#2c3e50'),
            tickfont=dict(size=12, color='#34495e'),
            tickformat='$,.0f'
        ),
        title=dict(
            text="Revenue Over Time",
            font=dict(size=18, color='#2c3e50', family="Arial Black"),
            x=0.5
        ),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        hovermode='x unified',
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("---")


def display_category_revenue_chart(df: pd.DataFrame) -> None:
    # Revenue by category bar chart with dynamic KPIs
    st.subheader("Category Performance")
    
    if df.empty:
        st.info("No data to display for the selected filters.")
        return

    # Calculate category-specific KPIs
    category_sales = (
        df.groupby("category")["revenue"].sum().reset_index(name="revenue")
    )
    
    top_category = category_sales.loc[category_sales["revenue"].idxmax()]
    lowest_category = category_sales.loc[category_sales["revenue"].idxmin()]
    num_categories = len(category_sales)
    avg_category_revenue = category_sales["revenue"].mean()
    
    # Display category KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Top Category", top_category["category"])
    with col2:
        st.metric("Lowest Category", lowest_category["category"])
    with col3:
        st.metric("Active Categories", f"{num_categories}")
    with col4:
        st.metric("Avg Category Revenue", f"${avg_category_revenue:,.2f}")
    
    # Display chart
    fig2 = px.bar(category_sales, x="category", y="revenue", title="Revenue by Category")
    
    # Enhanced styling
    fig2.update_traces(
        marker=dict(
            color='#3498db',
            line=dict(color='#2980b9', width=1.5),
            opacity=0.8
        ),
        hovertemplate='<b>Category:</b> %{x}<br><b>Revenue:</b> $%{y:,.2f}<extra></extra>'
    )
    
    fig2.update_layout(
        xaxis=dict(
            fixedrange=True,
            title="Category",
            title_font=dict(size=14, color='#2c3e50'),
            tickfont=dict(size=11, color='#34495e'),
            tickangle=45
        ),
        yaxis=dict(
            fixedrange=True,
            title="Revenue ($)",
            title_font=dict(size=14, color='#2c3e50'),
            tickfont=dict(size=12, color='#34495e'),
            tickformat='$,.0f'
        ),
        title=dict(
            text="Revenue by Category",
            font=dict(size=18, color='#2c3e50', family="Arial Black"),
            x=0.5
        ),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        margin=dict(l=60, r=60, t=80, b=100)
    )
    
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")


def display_pie_chart(df: pd.DataFrame) -> None:
    # Revenue distribution pie chart with dynamic KPIs
    st.subheader("Revenue Distribution")
    
    if df.empty:
        st.info("No data to display for the selected filters.")
        return

    # Calculate distribution-specific KPIs
    category_sales = (
        df.groupby("category")["revenue"].sum().reset_index(name="revenue")
    )
    
    total_revenue = category_sales["revenue"].sum()
    largest_share = category_sales["revenue"].max()
    largest_share_pct = (largest_share / total_revenue * 100)
    smallest_share = category_sales["revenue"].min()
    smallest_share_pct = (smallest_share / total_revenue * 100)
    
    # Calculate revenue concentration (top 3 categories)
    top_3_revenue = category_sales.nlargest(3, "revenue")["revenue"].sum()
    concentration_pct = (top_3_revenue / total_revenue * 100)
    
    # Display distribution KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Largest Share", f"{largest_share_pct:.1f}%")
    with col2:
        st.metric("Smallest Share", f"{smallest_share_pct:.1f}%")
    with col3:
        st.metric("Top 3 Concentration", f"{concentration_pct:.1f}%")
    with col4:
        revenue_spread = largest_share_pct - smallest_share_pct
        st.metric("Revenue Spread", f"{revenue_spread:.1f}%")
    
    # Display chart
    fig3 = px.pie(category_sales, values="revenue", names="category", title="Revenue Distribution by Category")
    
    # Enhanced styling with beautiful colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    fig3.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont=dict(size=12, color='white', family="Arial"),
        marker=dict(
            colors=colors,
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig3.update_layout(
        title=dict(
            text="Revenue Distribution by Category",
            font=dict(size=18, color='#2c3e50', family="Arial Black"),
            x=0.5
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=11, color='#34495e')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        margin=dict(l=60, r=120, t=80, b=60),
        dragmode=False
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("---")


def display_orders_chart(df: pd.DataFrame) -> None:
    # Orders by category bar chart with dynamic KPIs
    st.subheader("Order Volume Analysis")
    
    if df.empty:
        st.info("No data to display for the selected filters.")
        return

    # Calculate order-specific KPIs
    category_orders = (
        df.groupby("category")["order_id"].count().reset_index(name="orders")
    )
    
    # Calculate additional order metrics
    daily_orders = df.groupby(df["order_date"].dt.date)["order_id"].count()
    peak_day_orders = daily_orders.max()
    avg_daily_orders = daily_orders.mean()
    
    most_ordered_category = category_orders.loc[category_orders["orders"].idxmax()]
    least_ordered_category = category_orders.loc[category_orders["orders"].idxmin()]
    
    # Display order KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Most Ordered", most_ordered_category["category"])
    with col2:
        st.metric("Least Ordered", least_ordered_category["category"])
    with col3:
        st.metric("Peak Day Orders", f"{peak_day_orders:,}")
    with col4:
        st.metric("Avg Daily Orders", f"{avg_daily_orders:.1f}")
    
    # Display chart
    fig4 = px.bar(category_orders, x="category", y="orders", title="Number of Orders by Category")
    
    # Enhanced styling with gradient colors
    fig4.update_traces(
        marker=dict(
            color='#E74C3C',
            line=dict(color='#C0392B', width=1.5),
            opacity=0.8
        ),
        hovertemplate='<b>Category:</b> %{x}<br><b>Orders:</b> %{y:,}<extra></extra>'
    )
    
    fig4.update_layout(
        xaxis=dict(
            fixedrange=True,
            title="Category",
            title_font=dict(size=14, color='#2c3e50'),
            tickfont=dict(size=11, color='#34495e'),
            tickangle=45
        ),
        yaxis=dict(
            fixedrange=True,
            title="Number of Orders",
            title_font=dict(size=14, color='#2c3e50'),
            tickfont=dict(size=12, color='#34495e'),
            tickformat=','
        ),
        title=dict(
            text="Number of Orders by Category",
            font=dict(size=18, color='#2c3e50', family="Arial Black"),
            x=0.5
        ),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        margin=dict(l=60, r=60, t=80, b=100)
    )
    
    fig4.update_xaxes(showgrid=False)
    fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("---")


def display_charts(df: pd.DataFrame) -> None:
    # Legacy function - now replaced by individual chart functions
    pass


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

    # Calculate comprehensive business metrics
    total_sales = float(df["revenue"].sum())
    total_orders = int(df["order_id"].nunique())
    avg_order_value = total_sales / total_orders if total_orders else 0
    
    # Category analysis
    category_revenue = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    category_orders = df.groupby("category")["order_id"].count().sort_values(ascending=False)
    top_category = category_revenue.index[0] if not category_revenue.empty else "N/A"
    top_category_revenue = category_revenue.iloc[0] if not category_revenue.empty else 0
    category_share = (top_category_revenue / total_sales * 100) if total_sales > 0 else 0
    
    # Time-based analysis
    df['order_date'] = pd.to_datetime(df['order_date'])
    daily_sales = df.groupby(df["order_date"].dt.date)["revenue"].sum()
    monthly_sales = df.groupby(df["order_date"].dt.to_period('M'))["revenue"].sum()
    
    peak_day = daily_sales.idxmax() if not daily_sales.empty else "N/A"
    peak_day_revenue = daily_sales.max() if not daily_sales.empty else 0
    avg_daily_revenue = daily_sales.mean() if not daily_sales.empty else 0
    
    # Business performance metrics
    days_active = len(daily_sales)
    orders_per_day = total_orders / days_active if days_active > 0 else 0
    revenue_per_day = total_sales / days_active if days_active > 0 else 0
    
    # Product performance
    top_products = df.groupby("product_name")["revenue"].sum().sort_values(ascending=False).head(3)
    
    # Growth analysis (if sufficient data)
    if len(monthly_sales) >= 2:
        latest_month_revenue = monthly_sales.iloc[-1]
        previous_month_revenue = monthly_sales.iloc[-2]
        monthly_growth = ((latest_month_revenue - previous_month_revenue) / previous_month_revenue * 100) if previous_month_revenue > 0 else 0
    else:
        monthly_growth = 0
        latest_month_revenue = monthly_sales.iloc[0] if len(monthly_sales) > 0 else 0
    
    # Price analysis
    avg_product_price = df["price"].mean()
    price_std = df["price"].std()
    
    # Build comprehensive prompt for AI analysis
    prompt = f"""
    Analyze this e-commerce business data and provide a comprehensive business intelligence report with specific metrics, actionable insights, and strategic recommendations.

    FINANCIAL PERFORMANCE METRICS:
    - Total Revenue: ${total_sales:,.2f}
    - Total Orders: {total_orders:,}
    - Average Order Value (AOV): ${avg_order_value:,.2f}
    - Daily Revenue Run Rate: ${revenue_per_day:,.2f}
    - Average Daily Orders: {orders_per_day:.1f}
    - Monthly Growth Rate: {monthly_growth:+.1f}%
    - Current Month Revenue: ${latest_month_revenue:,.2f}

    CATEGORY PERFORMANCE ANALYSIS:
    - Top Category: {top_category} (${top_category_revenue:,.2f} - {category_share:.1f}% of total revenue)
    - Total Categories: {len(category_revenue)}
    - Category Revenue Distribution: {dict(category_revenue.head(3))}
    - Category Order Volume: {dict(category_orders.head(3))}

    OPERATIONAL METRICS:
    - Peak Sales Day: {peak_day} (${peak_day_revenue:,.2f})
    - Average Daily Revenue: ${avg_daily_revenue:,.2f}
    - Business Active Days: {days_active}
    - Average Product Price: ${avg_product_price:.2f}
    - Price Volatility (Std Dev): ${price_std:.2f}

    TOP PERFORMING PRODUCTS:
    {dict(top_products)}

    ANALYSIS REQUIREMENTS:
    1. Provide KPI analysis with specific percentage benchmarks
    2. Identify revenue concentration risks and opportunities
    3. Calculate customer acquisition efficiency metrics
    4. Suggest pricing optimization strategies
    5. Recommend inventory management improvements
    6. Forecast growth potential based on current trends
    7. Identify seasonal patterns and business cycles
    8. Provide competitive positioning insights
    9. Suggest marketing budget allocation by category
    10. Recommend operational efficiency improvements

    Format as a professional business analysis report with numbered sections and specific dollar amounts and percentages throughout.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
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
    
    # Display all charts on the same page with their own KPIs
    display_revenue_chart(filtered_df)
    display_category_revenue_chart(filtered_df)
    display_pie_chart(filtered_df)
    display_orders_chart(filtered_df)

    # AI Insights Button
    if st.sidebar.button("Generate AI Insights"):
        with st.spinner():
            insights = generate_ai_insights(filtered_df)
        st.subheader("AI-Generated Insights")
        st.write(insights)


if __name__ == "__main__":
    main()















##Ansh Gopinath