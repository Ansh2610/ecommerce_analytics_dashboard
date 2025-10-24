"""
Utility script to generate mock e-commerce data.

Creates a CSV file with synthetic order records using Faker for realistic data.
Usage: python generate_data.py --orders 2000 --seed 123
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from faker import Faker


def generate_dataframe(n_orders: int = 10000, seed: int = 42) -> pd.DataFrame:
    # Generate synthetic orders using Faker for realistic product names and pricing
    fake = Faker()
    fake.seed_instance(seed)
    np.random.seed(seed)
    
    # Product categories with realistic names
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
    
    # Price ranges per category
    price_ranges = {
        "Electronics": (50, 2000),
        "Clothing": (15, 500),
        "Home & Garden": (25, 800),
        "Sports & Outdoors": (20, 600),
        "Books & Media": (5, 200),
        "Health & Beauty": (12, 200),
        "Automotive": (30, 1200)
    }
    
    records: list[list] = []
    
    for i in range(1, n_orders + 1):
        # Select category and product
        category = fake.random_element(elements=list(categories_products.keys()))
        product = fake.random_element(elements=categories_products[category])
        
        # Price based on category range
        min_price, max_price = price_ranges[category]
        price = round(fake.random.uniform(min_price, max_price), 2)
        
        # Most orders are 1-2 items
        quantity = fake.random_element(elements=[1, 1, 1, 2, 2, 3, 4, 5])
        
        # Order date between 2015-2025 (10 years of data)
        order_date = fake.date_time_between(start_date='-10y', end_date='now')
        
        revenue = round(price * quantity, 2)
        
        records.append([i, category, product, price, quantity, order_date, revenue])
    
    df = pd.DataFrame(
        records,
        columns=["order_id", "category", "product_name", "price", "quantity", "order_date", "revenue"],
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a mock e‑commerce dataset.")
    parser.add_argument("--orders", type=int, default=1000, help="Number of orders to generate")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to ensure reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/data.csv",
        help="Output CSV path relative to the project root",
    )
    args = parser.parse_args()

    df = generate_dataframe(n_orders=args.orders, seed=args.seed)

    # Save to CSV file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows and wrote to {output_path}")


if __name__ == "__main__":
    main()