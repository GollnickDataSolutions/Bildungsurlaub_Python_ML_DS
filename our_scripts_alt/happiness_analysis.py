#!/usr/bin/env python3
"""
Analysis script for the happiness dataset.
This script analyzes happiness scores by country and gender from the happiness.csv file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(file_path):
    """Load the happiness dataset"""
    try:
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data(df):
    """Explore the structure and content of the dataset"""
    print("=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nFirst few rows:")
    print(df.head(10))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nUnique countries:")
    print(f"Total countries: {df['Country'].nunique()}")
    print(df['Country'].unique())

def analyze_happiness_by_country(df):
    """Analyze happiness scores by country"""
    print("\n=== HAPPINESS ANALYSIS BY COUNTRY ===")

    # Filter out aggregated rows (rows with empty Country values)
    country_data = df[df['Country'].notna() & (df['Country'] != '')]

    print(f"Number of countries in dataset: {country_data['Country'].nunique()}")

    # Calculate mean happiness by country
    country_avg = country_data.groupby('Country')['Mean'].mean().sort_values(ascending=False)
    print("\nAverage happiness scores by country (descending):")
    for country, score in country_avg.items():
        print(f"  {country}: {score:.2f}")

    return country_avg

def analyze_happiness_by_gender(df):
    """Analyze happiness scores by gender"""
    print("\n=== HAPPINESS ANALYSIS BY GENDER ===")

    # Filter out aggregated rows (rows with empty Country values)
    gender_data = df[df['Country'].notna() & (df['Country'] != '')]

    # Calculate mean happiness by gender
    gender_avg = gender_data.groupby('Gender')['Mean'].mean()
    print("\nAverage happiness scores by gender:")
    for gender, score in gender_avg.items():
        print(f"  {gender}: {score:.2f}")

    return gender_avg

def visualize_data(df):
    """Create visualizations for the dataset"""
    print("\n=== CREATING VISUALIZATIONS ===")

    # Create directory for plots if it doesn't exist
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)

    # Filter out aggregated rows
    clean_data = df[df['Country'].notna() & (df['Country'] != '')]

    # 1. Happiness scores by country (top 10)
    plt.figure(figsize=(12, 8))
    country_avg = clean_data.groupby('Country')['Mean'].mean().sort_values(ascending=False).head(10)
    bars = plt.bar(range(len(country_avg)), country_avg.values)
    plt.xlabel('Country')
    plt.ylabel('Average Happiness Score')
    plt.title('Top 10 Countries by Average Happiness Score')
    plt.xticks(range(len(country_avg)), country_avg.index, rotation=45, ha='right')

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, country_avg.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{value:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(plot_dir / 'happiness_by_country.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved: plots/happiness_by_country.png")

    # 2. Happiness scores by gender
    plt.figure(figsize=(8, 6))
    gender_avg = clean_data.groupby('Gender')['Mean'].mean()
    bars = plt.bar(range(len(gender_avg)), gender_avg.values)
    plt.xlabel('Gender')
    plt.ylabel('Average Happiness Score')
    plt.title('Average Happiness Score by Gender')
    plt.xticks(range(len(gender_avg)), gender_avg.index)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, gender_avg.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{value:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(plot_dir / 'happiness_by_gender.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved: plots/happiness_by_gender.png")

    # 3. Happiness scores by country and gender (for top 5 countries)
    top_countries = clean_data.groupby('Country')['Mean'].mean().sort_values(ascending=False).head(5).index

    plt.figure(figsize=(12, 8))
    for i, country in enumerate(top_countries):
        country_data = clean_data[clean_data['Country'] == country]
        gender_means = country_data.groupby('Gender')['Mean'].mean()

        plt.subplot(2, 3, i+1)
        bars = plt.bar(range(len(gender_means)), gender_means.values)
        plt.title(f'{country}')
        plt.ylabel('Happiness Score')
        plt.xticks(range(len(gender_means)), gender_means.index)

        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, gender_means.values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(plot_dir / 'happiness_by_country_gender.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved: plots/happiness_by_country_gender.png")

def main():
    """Main analysis function"""
    print("Starting happiness dataset analysis...")

    # Load data
    file_path = "our_scripts/happiness.csv"
    df = load_data(file_path)

    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Explore data
    explore_data(df)

    # Analyze by country
    country_avg = analyze_happiness_by_country(df)

    # Analyze by gender
    gender_avg = analyze_happiness_by_gender(df)

    # Create visualizations
    visualize_data(df)

    print("\n=== ANALYSIS COMPLETE ===")
    print("Visualizations saved to 'plots' directory")

if __name__ == "__main__":
    main()