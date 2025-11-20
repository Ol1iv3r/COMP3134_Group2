import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import networkx as nx
from collections import Counter
import warnings
import os  # Import os module

warnings.filterwarnings('ignore')

# Set font and graph styles
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_and_preprocess_data(file_path):
    """Load and preprocess data"""
    # Use the passed file path
    df = pd.read_csv(file_path)


    # Data preprocessing
    # Convert date format
    df['Invoice date'] = pd.to_datetime(df['Invoice date'], format='%d/%m/%Y')

    # Extract year and month information
    df['Year'] = df['Invoice date'].dt.year
    df['Month'] = df['Invoice date'].dt.month
    df['YearMonth'] = df['Invoice date'].dt.to_period('M')

    # Calculate number of products per order
    df['Product_count'] = df['Product id list'].apply(lambda x: len(str(x).split(',')))

    return df

def explode_product_data(df):
    """Split product ID list into separate rows"""
    # Copy product ID column and clean
    df_products = df.copy()
    df_products['Product id list'] = df_products['Product id list'].astype(str).str.replace('"', '')

    # Split product IDs
    product_exploded = df_products.assign(
        Product_id=df_products['Product id list'].str.split(',')
    ).explode('Product_id')

    # Clean product IDs
    product_exploded['Product_id'] = product_exploded['Product_id'].str.strip()
    product_exploded = product_exploded[product_exploded['Product_id'] != '']

    return product_exploded

def plot_sales_trends(df):
    """Plot sales trends"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sales Trend Analysis', fontsize=16, fontweight='bold')

    # 1. Monthly sales trend
    monthly_trend = df.groupby('YearMonth').size()
    monthly_trend.index = monthly_trend.index.astype(str)
    axes[0,0].plot(monthly_trend.index, monthly_trend.values, marker='o', linewidth=2, markersize=4)
    axes[0,0].set_title('Monthly Order Trend')
    axes[0,0].set_xlabel('Year-Month')
    axes[0,0].set_ylabel('Number of Orders')
    axes[0,0].tick_params(axis='x', rotation=45)

    # 2. Monthly sales heatmap
    monthly_pivot = df.groupby(['Year', 'Month']).size().unstack(fill_value=0)
    sns.heatmap(monthly_pivot, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0,1])
    axes[0,1].set_title('Monthly Sales Heatmap')

    # 3. Monthly trend by shopping mall
    mall_monthly = df.groupby(['YearMonth', 'Shopping mall']).size().unstack(fill_value=0)
    mall_monthly.index = mall_monthly.index.astype(str)
    mall_monthly.plot(ax=axes[1,0], linewidth=2)
    axes[1,0].set_title('Monthly Sales Trend by Shopping Mall')
    axes[1,0].set_xlabel('Year-Month')
    axes[1,0].set_ylabel('Number of Orders')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 4. Distribution of products per order
    product_count_dist = df['Product_count'].value_counts().sort_index()
    axes[1,1].bar(product_count_dist.index, product_count_dist.values, color='skyblue', alpha=0.7)
    axes[1,1].set_title('Distribution of Products per Order')
    axes[1,1].set_xlabel('Number of Products per Order')
    axes[1,1].set_ylabel('Number of Orders')

    plt.tight_layout()
    plt.show(block=False)  # Non-blocking mode

def plot_mall_performance(df):
    """Plot shopping mall performance analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Shopping Mall Performance Analysis', fontsize=16, fontweight='bold')

    # 1. Order count by shopping mall
    mall_orders = df['Shopping mall'].value_counts()
    axes[0,0].bar(mall_orders.index, mall_orders.values, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
    axes[0,0].set_title('Order Count by Shopping Mall')
    axes[0,0].set_ylabel('Number of Orders')

    # 2. Pie chart showing market share
    axes[0,1].pie(mall_orders.values, labels=mall_orders.index, autopct='%1.1f%%',
                  colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
    axes[0,1].set_title('Market Share by Shopping Mall')

    # 3. Average products per order by mall
    mall_avg_products = df.groupby('Shopping mall')['Product_count'].mean()
    axes[1,0].bar(mall_avg_products.index, mall_avg_products.values, color='lightcoral', alpha=0.7)
    axes[1,0].set_title('Average Products per Order by Mall')
    axes[1,0].set_ylabel('Average Number of Products')

    # 4. Monthly growth trend by mall (relative to first period)
    mall_growth = df.groupby(['YearMonth', 'Shopping mall']).size().unstack(fill_value=0)
    # Calculate growth rate relative to first month
    growth_rates = (mall_growth / mall_growth.iloc[0] * 100) - 100
    growth_rates.index = growth_rates.index.astype(str)

    for mall in growth_rates.columns:
        axes[1,1].plot(growth_rates.index, growth_rates[mall], marker='o', label=mall, markersize=3)

    axes[1,1].set_title('Sales Growth Trend by Mall (Relative to First Period)')
    axes[1,1].set_xlabel('Year-Month')
    axes[1,1].set_ylabel('Growth Rate (%)')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].legend()
    axes[1,1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show(block=False)  # Non-blocking mode

def plot_product_analysis(product_exploded):
    """Plot product analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Product Analysis', fontsize=16, fontweight='bold')

    # 1. Top 20 best-selling products
    top_products = product_exploded['Product_id'].value_counts().head(20)
    axes[0,0].barh(range(len(top_products)), top_products.values)
    axes[0,0].set_yticks(range(len(top_products)))
    axes[0,0].set_yticklabels(top_products.index)
    axes[0,0].set_title('Top 20 Best-Selling Products')
    axes[0,0].set_xlabel('Sales Count')

    # 2. Product purchase frequency distribution - fix blank issue
    product_freq = product_exploded['Product_id'].value_counts()
    # Limit display range, only show products with purchase count <= 50
    freq_dist = product_freq.value_counts().sort_index()
    max_display = min(50, freq_dist.index.max())
    filtered_freq_dist = freq_dist[freq_dist.index <= max_display]

    axes[0,1].bar(filtered_freq_dist.index, filtered_freq_dist.values, alpha=0.7, color='lightgreen')
    axes[0,1].set_title('Product Purchase Frequency Distribution')
    axes[0,1].set_xlabel('Purchase Count')
    axes[0,1].set_ylabel('Number of Products')
    axes[0,1].set_xlim(0, max_display + 2)
    axes[0,1].set_xticks(range(0, max_display + 1, max(1, max_display//10)))

    # 3. Best-selling products comparison by mall
    # First calculate the best-selling product for each mall
    mall_top_products = product_exploded.groupby('Shopping mall')['Product_id'].apply(
        lambda x: x.value_counts().head(1)
    )

    # Reset index and rename columns
    mall_top_products = mall_top_products.reset_index()
    mall_top_products.columns = ['Shopping mall', 'Product_id', 'Sales_Count']

    # Create bar chart showing best-selling product for each mall
    mall_labels = []
    product_counts = []

    for idx, row in mall_top_products.iterrows():
        mall_labels.append(f"{row['Shopping mall']}\n({row['Product_id']})")
        product_counts.append(row['Sales_Count'])

    axes[1,0].bar(mall_labels, product_counts, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
    axes[1,0].set_title('Best-Selling Product by Mall')
    axes[1,0].set_ylabel('Sales Count')
    axes[1,0].tick_params(axis='x', rotation=45)

    # 4. Product combination purchase patterns (most common product combination sizes)
    product_combinations = product_exploded.groupby('Invoice no').size().value_counts().sort_index()
    axes[1,1].bar(product_combinations.index, product_combinations.values, color='coral', alpha=0.7)
    axes[1,1].set_title('Product Combination Purchase Patterns')
    axes[1,1].set_xlabel('Number of Product Types per Purchase')
    axes[1,1].set_ylabel('Number of Orders')

    plt.tight_layout()
    plt.show(block=False)  # Non-blocking mode

def plot_customer_analysis(df, product_exploded):
    """Plot customer analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Customer Behavior Analysis', fontsize=16, fontweight='bold')

    # 1. Customer purchase frequency distribution
    customer_freq = df['Customer id'].value_counts()
    freq_dist = customer_freq.value_counts().sort_index()
    axes[0,0].bar(freq_dist.index, freq_dist.values, alpha=0.7, color='lightblue')
    axes[0,0].set_title('Customer Purchase Frequency Distribution')
    axes[0,0].set_xlabel('Purchase Count')
    axes[0,0].set_ylabel('Number of Customers')

    # 2. Top 15 most active customers
    top_customers = customer_freq.head(15)
    axes[0,1].barh(range(len(top_customers)), top_customers.values)
    axes[0,1].set_yticks(range(len(top_customers)))
    axes[0,1].set_yticklabels([f'Customer {cid}' for cid in top_customers.index])
    axes[0,1].set_title('Top 15 Most Active Customers')
    axes[0,1].set_xlabel('Number of Orders')

    # 3. Customer value analysis (based on number of products purchased)
    customer_value = product_exploded.groupby('Customer id').size().sort_values(ascending=False).head(15)
    axes[1,0].barh(range(len(customer_value)), customer_value.values, color='orange', alpha=0.7)
    axes[1,0].set_yticks(range(len(customer_value)))
    axes[1,0].set_yticklabels([f'Customer {cid}' for cid in customer_value.index])
    axes[1,0].set_title('Customer Value Analysis (Top 15 by Total Products Purchased)')
    axes[1,0].set_xlabel('Total Products Purchased')

    # 4. Customer loyalty analysis (by mall)
    customer_mall_loyalty = df.groupby('Customer id')['Shopping mall'].nunique()
    loyalty_dist = customer_mall_loyalty.value_counts().sort_index()
    axes[1,1].pie(loyalty_dist.values, labels=[f'{i} Malls' for i in loyalty_dist.index],
                  autopct='%1.1f%%', colors=['gold', 'lightcoral', 'lightgreen', 'lightblue'])
    axes[1,1].set_title('Customer Mall Loyalty Distribution')

    plt.tight_layout()
    plt.show(block=False)  # Non-blocking mode

def create_product_network(product_exploded, top_n=15):
    """Create product association network graph (optimized layout and readability)"""
    # Get top N most popular products
    top_products = product_exploded['Product_id'].value_counts().head(top_n).index.tolist()

    # Filter data to only include popular products
    filtered_data = product_exploded[product_exploded['Product_id'].isin(top_products)]

    # Create co-occurrence matrix
    product_pairs = filtered_data.groupby('Invoice no')['Product_id'].apply(
        lambda x: list(x) if len(x) > 1 else None
    ).dropna().reset_index()

    # Build network graph
    G = nx.Graph()

    # Add nodes (products)
    for product in top_products:
        G.add_node(product, size=filtered_data[filtered_data['Product_id'] == product].shape[0])

    # Add edges (co-purchase relationships)
    edge_weights = {}
    for _, row in product_pairs.iterrows():
        products = row['Product_id']
        for i in range(len(products)):
            for j in range(i+1, len(products)):
                edge = tuple(sorted([products[i], products[j]]))
                edge_weights[edge] = edge_weights.get(edge, 0) + 1

    # Only add edges with weight >= 3 to reduce clutter
    for (u, v), weight in edge_weights.items():
        if weight >= 3 and u in G.nodes and v in G.nodes:
            G.add_edge(u, v, weight=weight)

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    # Create new graph, keep only largest connected component
    if G.nodes():
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # If no edges, return directly
    if G.number_of_edges() == 0:
        print("No significant product associations found (all connections have weight < 3).")
        return

    plt.figure(figsize=(20, 15))

    # Calculate node sizes (based on product popularity)
    node_sizes = [G.nodes[node]['size'] * 20 for node in G.nodes()]  # Slightly increase node size

    # Calculate edge widths (based on co-occurrence frequency)
    edge_widths = [G[u][v]['weight'] * 0.8 for u, v in G.edges()]  # Slightly thicken edges

    # Use Spring layout to better disperse nodes, add seed to fix layout
    pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42)  # Optimize k value and iterations to reduce overlap

    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                           alpha=0.8, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5,
                           edge_color='gray', style='solid')

    # Add labels for all nodes, but use smaller font and adjust position to reduce overlap
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold',
                            verticalalignment='center', horizontalalignment='center')

    plt.title(f'Product Association Network (TOP{top_n} Products, connections ≥3)\n'
              'Node size = product popularity | Line thickness = co-purchase frequency',
              fontsize=16, pad=20)
    plt.axis('off')

    # Add legend description
    plt.figtext(0.5, 0.01, 'Products connected by lines are frequently purchased together',
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Add edge weight labels (only label edges with weight > 5, adjust position to avoid overlap)
    for (u, v, d) in G.edges(data=True):
        if d['weight'] > 5:
            mid_x = (pos[u][0] + pos[v][0]) / 2 + 0.02  # Slight offset to reduce overlap
            mid_y = (pos[u][1] + pos[v][1]) / 2 + 0.02
            plt.text(mid_x, mid_y, str(d['weight']), fontsize=8,
                     bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=0.2))

    plt.tight_layout()
    plt.show(block=False)  # Non-blocking mode

def main():
    """Main function"""
    # File path
    file_path = 'group_2/sales_2.csv'

    print("Loading and preprocessing data...")
    # Load data
    df = load_and_preprocess_data(file_path)

    print("Processing product data...")
    # Split product data
    product_exploded = explode_product_data(df)

    print("Generating sales trend analysis charts...")
    # Sales trend analysis
    plot_sales_trends(df)

    print("Generating mall performance analysis charts...")
    # Mall performance analysis
    plot_mall_performance(df)

    print("Generating product analysis charts...")
    # Product analysis
    plot_product_analysis(product_exploded)

    print("Generating customer analysis charts...")
    # Customer analysis
    plot_customer_analysis(df, product_exploded)

    # Product association network chart (commented out by default to improve readability during initial runs)
    #create_product_network(product_exploded, top_n=15)  # Reduce node count for better readability

    # Print some basic statistics
    print("\n=== Dataset Basic Statistics ===")
    print(f"Total number of orders: {len(df)}")
    print(f"Time range: {df['Invoice date'].min()} to {df['Invoice date'].max()}")
    print(f"Number of unique customers: {df['Customer id'].nunique()}")
    print(f"Number of unique products: {product_exploded['Product_id'].nunique()}")
    print(f"Number of shopping malls: {df['Shopping mall'].nunique()}")
    print(f"Average products per order: {df['Product_count'].mean():.2f}")

    # Shopping mall statistics
    print("\n=== Shopping Mall Statistics ===")
    mall_stats = df.groupby('Shopping mall').agg({
        'Invoice no': 'count',
        'Product_count': 'mean',
        'Customer id': 'nunique'
    }).rename(columns={'Invoice no': 'Order Count', 'Product_count': 'Avg Products', 'Customer id': 'Customer Count'})
    print(mall_stats)

    # Keep all chart windows open until user closes them
    plt.show()

if __name__ == "__main__":
    main()