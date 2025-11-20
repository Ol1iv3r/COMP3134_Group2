import os
import pandas as pd
import numpy as np
import matplotlib

# Attempt to use different backends
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('Agg')  # Fallback to non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse

CSV_PATH = "group_2/products_2.csv"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path)
    return df


def comprehensive_product_analysis(df: pd.DataFrame):
    """Comprehensive Product Analysis - similar to customer analysis structure"""

    def price_group(price):
        """Group prices into categories"""
        if price < 20:
            return "0-20"
        elif price < 50:
            return "20-50"
        elif price < 100:
            return "50-100"
        elif price < 200:
            return "100-200"
        else:
            return "200+"

    print("=== Basic Product Data Analysis ===")
    print(f"Total number of products: {len(df)}")
    print(f"Category distribution:\n{df['Category'].value_counts()}")
    print(f"\nPrice statistics:")
    print(df['Price'].describe())

    # Price group analysis
    df['Price_Group'] = df['Price'].apply(price_group)

    # Visualization results
    plt.figure(figsize=(15, 10))

    # 1. Category distribution pie chart
    plt.subplot(2, 2, 1)
    category_counts = df['Category'].value_counts()
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Product Category Distribution')

    # 2. Price group distribution
    plt.subplot(2, 2, 2)
    price_groups = df['Price_Group'].value_counts().sort_index()
    plt.bar(price_groups.index, price_groups.values, color='skyblue')
    plt.title('Price Group Distribution')
    plt.xlabel('Price Range')
    plt.ylabel('Number of Products')
    for i, v in enumerate(price_groups.values):
        plt.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

    # 3. Category vs. Price group
    plt.subplot(2, 2, 3)
    cross_tab = pd.crosstab(df['Category'], df['Price_Group'])
    cross_tab.plot(kind='bar', ax=plt.gca())
    plt.title('Price Distribution by Category')
    plt.xticks(rotation=45)
    plt.legend(title='Price Group')

    # 4. Average price per category
    plt.subplot(2, 2, 4)
    avg_price_by_category = df.groupby('Category')['Price'].mean().sort_values(ascending=False)
    plt.bar(avg_price_by_category.index, avg_price_by_category.values, color='lightgreen')
    plt.title('Average Price by Category')
    plt.xticks(rotation=45)
    plt.ylabel('Average Price')
    for i, v in enumerate(avg_price_by_category.values):
        plt.text(i, v, f'${v:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Add functionality to save the graph
    try:
        plt.savefig('./product_analysis.png')
        print("Graph saved as product_analysis.png")
    except Exception as e:
        print(f"Could not save graph: {e}")

    plt.show()

    # Key insights summary
    print("\n" + "=" * 60)
    print("Product Pricing Strategy Key Insights")
    print("=" * 60)

    # Price distribution by category
    print("\nPrice distribution by category:")
    category_price_stats = df.groupby('Category')['Price'].agg(['count', 'mean', 'median', 'std']).round(2)
    print(category_price_stats)

    # Generate product strategy recommendations
    print("\nProduct Strategy Recommendations:")

    for category in sorted(df['Category'].unique()):
        category_data = df[df['Category'] == category]
        avg_price = category_data['Price'].mean()
        price_std = category_data['Price'].std()
        size = len(category_data)

        print(f"\n{category} category ({size} products, {size / len(df) * 100:.1f}%):")
        print(f"  Average price: ${avg_price:.2f}")
        print(f"  Price variability: ${price_std:.2f} (std)")

        # Provide strategy recommendations based on price analysis
        if avg_price < 30:
            print("  Recommended strategy: Economy segment - focus on volume sales, bundle offers")
        elif avg_price < 80:
            print("  Recommended strategy: Mid-range segment - balance quality and affordability")
        elif avg_price < 150:
            print("  Recommended strategy: Premium segment - emphasize quality and features")
        else:
            print("  Recommended strategy: Luxury segment - focus on exclusivity and premium experience")

    # In-depth price group analysis
    print("\n" + "=" * 60)
    print("Price Group In-depth Analysis")
    print("=" * 60)

    for price_group in sorted(df['Price_Group'].unique()):
        group_data = df[df['Price_Group'] == price_group]
        category_dist = group_data['Category'].value_counts(normalize=True).head(3) * 100

        print(f"\n{price_group} price range:")
        print(f"  Number of products: {len(group_data)}")
        print(f"  Top categories: {', '.join([f'{cat}({pct:.1f}%)' for cat, pct in category_dist.items()])}")

        if price_group == "0-20":
            print("  Characteristics: Budget products, high volume potential")
        elif price_group == "20-50":
            print("  Characteristics: Value segment, competitive pricing")
        elif price_group == "50-100":
            print("  Characteristics: Mid-market, quality focus")
        elif price_group == "100-200":
            print("  Characteristics: Premium products, feature-rich")
        else:
            print("  Characteristics: High-end products, exclusive market")


def plot_category_counts(df: pd.DataFrame):
    """Bar chart of product counts by category"""
    counts = df["Category"].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values, color=plt.cm.viridis(np.linspace(0, 1, len(counts))))
    plt.title("Number of Products per Category", fontsize=16, pad=20)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    for i, v in enumerate(counts.values):
        plt.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()

    # Add functionality to save the graph
    try:
        plt.savefig('./category_counts.png')
        print("Graph saved as category_counts.png")
    except Exception as e:
        print(f"Could not save graph: {e}")

    plt.show()


def plot_price_cum_rel_freq(df: pd.DataFrame, category: str = None):
    """Price cumulative relative frequency table"""
    if category:
        subset = df[df['Category'] == category]
        title = f"Price Distribution Table for {category} (10 Classes)"
    else:
        subset = df
        title = "Overall Price Distribution Table (10 Classes)"

    prices = subset['Price'].dropna()
    if len(prices) == 0:
        print(f"No data for {category or 'Overall'}")
        return

    freq, bin_edges = np.histogram(prices, bins=10)
    rel_freq = freq / len(prices)
    cum_freq = np.cumsum(freq)
    cum_rel_freq = np.cumsum(rel_freq)
    bin_labels = [f"${bin_edges[i]:.2f} - ${bin_edges[i + 1]:.2f}" for i in range(len(bin_edges) - 1)]

    # Create table display
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    cellText = [[label, str(f), str(cf), f"{crf:.4f}"] for label, f, cf, crf in
                zip(bin_labels, freq, cum_freq, cum_rel_freq)]
    colLabels = ['Price Range', 'Frequency', 'Cumulative Frequency', 'Cumulative Relative Frequency']
    table = ax.table(cellText=cellText, colLabels=colLabels, loc='center', cellLoc='center',
                     colWidths=[0.3, 0.15, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Set header style
    for i in range(len(colLabels)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()

    # Add functionality to save the graph
    try:
        if category:
            filename = f'./price_table_{category.replace(" ", "_")}.png'
        else:
            filename = './price_table_overall.png'
        plt.savefig(filename)
        print(f"Graph saved as {filename}")
    except Exception as e:
        print(f"Could not save graph: {e}")

    plt.show()


def show_descriptive_stats(df: pd.DataFrame, category: str = None):
    """Descriptive statistics table"""
    if category:
        subset = df[df['Category'] == category].copy()
        title = f"Descriptive Statistics for Price - {category}"
    else:
        subset = df.copy()
        title = "Descriptive Statistics for Price - Overall"

    prices = subset['Price'].dropna()
    if len(prices) == 0:
        print(f"No price data for {category or 'overall'}.")
        return

    stats = {
        "Count": len(prices),
        "Mean": f"${prices.mean():.2f}",
        "Median": f"${prices.median():.2f}",
        "Std. Deviation": f"${prices.std():.2f}",
        "Variance": f"${prices.var():.2f}",
        "Min": f"${prices.min():.2f}",
        "Max": f"${prices.max():.2f}",
        "Range": f"${prices.max() - prices.min():.2f}"
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    stats_data = [[key, value] for key, value in stats.items()]
    col_labels = ['Statistic', 'Value']
    table = ax.table(cellText=stats_data, colLabels=col_labels, loc='center', cellLoc='left', colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Set header style
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    # Add functionality to save the graph
    try:
        if category:
            filename = f'./stats_{category.replace(" ", "_")}.png'
        else:
            filename = './stats_overall.png'
        plt.savefig(filename)
        print(f"Graph saved as {filename}")
    except Exception as e:
        print(f"Could not save graph: {e}")

    plt.show()


# Add missing function definitions
def show_linear_regression(df: pd.DataFrame):
    """Example of linear regression analysis"""
    print("Performing linear regression analysis (Price vs Category)...")
    
    # Prepare data: Encode Category
    categories = df['Category'].unique()
    cat_to_id = {cat: i for i, cat in enumerate(categories)}
    df_analysis = df.copy()
    df_analysis['Category_ID'] = df_analysis['Category'].map(cat_to_id)
    
    X = df_analysis[['Category_ID']]
    y = df_analysis['Price']
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    
    print(f"Linear Regression Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Coefficient: {model.coef_[0]:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Price')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted Price')
    
    plt.title(f'Linear Regression: Price vs Category (R²: {r2:.4f})')
    plt.xlabel('Category')
    plt.ylabel('Price')
    plt.xticks(ticks=list(cat_to_id.values()), labels=list(cat_to_id.keys()), rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Add functionality to save the graph
    try:
        plt.savefig('./linear_regression.png')
        print("Graph saved as linear_regression.png")
    except Exception as e:
        print(f"Could not save graph: {e}")
        
    plt.show()


def show_logistic_regression(df: pd.DataFrame, category: str):
    """Example of logistic regression analysis"""
    print(f"Performing logistic regression analysis for {category}...")
    
    # Prepare data: Target is 1 if product is in category, 0 otherwise
    df_analysis = df.copy()
    df_analysis['Target'] = (df_analysis['Category'] == category).astype(int)
    
    if len(df_analysis['Target'].unique()) < 2:
        print("Error: Not enough classes for logistic regression (need both positive and negative samples).")
        return

    X = df_analysis[['Price']]
    y = df_analysis['Target']
    
    # Train model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Evaluation
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Logistic Regression Results for '{category}':")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Coefficient: {model.coef_[0][0]:.4f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of actual data
    plt.scatter(df_analysis['Price'], df_analysis['Target'], color='blue', alpha=0.5, label='Data Points')
    
    # Sigmoid curve
    X_test = np.linspace(df_analysis['Price'].min(), df_analysis['Price'].max(), 300).reshape(-1, 1)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    plt.plot(X_test, y_prob, color='red', linewidth=2, label='Probability Curve')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f'Logistic Regression: Probability of being {category} by Price')
    plt.xlabel('Price')
    plt.ylabel(f'Probability (Is {category})')
    plt.legend()
    plt.tight_layout()
    
    # Add functionality to save the graph
    try:
        filename = f'./logistic_regression_{category.replace(" ", "_")}.png'
        plt.savefig(filename)
        print(f"Graph saved as {filename}")
    except Exception as e:
        print(f"Could not save graph: {e}")
        
    plt.show()


def show_clustering(df: pd.DataFrame):
    """Example of clustering analysis"""
    print("Performing K-Means clustering analysis on Price...")
    
    # Prepare data
    X = df[['Price']]
    
    # Elbow method to find optimal k (simplified for demo, using k=3)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_analysis = df.copy()
    df_analysis['Cluster'] = kmeans.fit_predict(X)
    
    centers = kmeans.cluster_centers_
    print(f"K-Means Clustering (k={k}):")
    print(f"  Cluster Centers (Price): {', '.join([f'${c[0]:.2f}' for c in sorted(centers)])}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    for i in range(k):
        cluster_data = df_analysis[df_analysis['Cluster'] == i]
        plt.scatter(cluster_data.index, cluster_data['Price'], 
                    color=colors[i % len(colors)], 
                    alpha=0.6, 
                    label=f'Cluster {i}')
        
    # Plot centers
    for i, center in enumerate(centers):
        plt.axhline(y=center[0], color=colors[i % len(colors)], linestyle='--', alpha=0.5)
        
    plt.title(f'K-Means Clustering by Price (k={k})')
    plt.xlabel('Product Index')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    
    # Add functionality to save the graph
    try:
        plt.savefig('./clustering_analysis.png')
        print("Graph saved as clustering_analysis.png")
    except Exception as e:
        print(f"Could not save graph: {e}")
        
    plt.show()



def main():
    # Check if data file exists
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        print("Please make sure the file exists and the path is correct.")
        return

    try:
        df = load_data(CSV_PATH)
        print(f"Loaded {len(df)} products.")
        categories = sorted(df['Category'].unique())
        print(f"Available categories: {', '.join(categories)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    while True:
        print("\n" + "=" * 40)
        print("Product Analysis Menu")
        print("=" * 40)
        print("1. Comprehensive Product Analysis (Overview)")
        print("2. View Product Counts by Category")
        print("3. View Price Distribution Tables")
        print("4. View Descriptive Statistics")
        print("5. Perform Linear Regression Analysis")
        print("6. Perform Logistic Regression Analysis")
        print("7. Perform Clustering Analysis")
        print("8. Quit")

        try:
            choice = int(input("Enter your choice (1-8): "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if choice == 1:
            comprehensive_product_analysis(df)
        elif choice == 2:
            plot_category_counts(df)
        elif choice == 3:
            print("\n=== Price Table Sub-Menu ===")
            print("1. Overall")
            for i, cat in enumerate(categories, 2):
                print(f"{i}. {cat}")
            try:
                sub_choice = int(input(f"Enter your choice (1-{len(categories) + 1}): "))
            except ValueError:
                print("Invalid input.")
                continue
            if sub_choice == 1:
                plot_price_cum_rel_freq(df)
            elif 2 <= sub_choice <= len(categories) + 1:
                plot_price_cum_rel_freq(df, categories[sub_choice - 2])
            else:
                print("Invalid choice.")
        elif choice == 4:
            print("\n=== Descriptive Statistics Sub-Menu ===")
            print("1. Overall")
            for i, cat in enumerate(categories, 2):
                print(f"{i}. {cat}")
            try:
                sub_choice = int(input(f"Enter your choice (1-{len(categories) + 1}): "))
            except ValueError:
                print("Invalid input.")
                continue
            if sub_choice == 1:
                show_descriptive_stats(df)
            elif 2 <= sub_choice <= len(categories) + 1:
                show_descriptive_stats(df, categories[sub_choice - 2])
            else:
                print("Invalid choice.")
        elif choice == 5:
            show_linear_regression(df)
        elif choice == 6:
            print("\n=== Logistic Regression Sub-Menu ===")
            for i, cat in enumerate(categories, 1):
                print(f"{i}. Is it {cat}?")
            try:
                sub_choice = int(input(f"Enter your choice (1-{len(categories)}): "))
            except ValueError:
                print("Invalid input.")
                continue
            if 1 <= sub_choice <= len(categories):
                show_logistic_regression(df, categories[sub_choice - 1])
            else:
                print("Invalid choice.")
        elif choice == 7:
            show_clustering(df)
        elif choice == 8:
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()