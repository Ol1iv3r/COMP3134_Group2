import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CustomerSegmentation:
    def __init__(self, customers_path, products_path, sales_path):
        """
        Initialize the Customer Segmentation analysis
        """
        self.customers_path = customers_path
        self.products_path = products_path
        self.sales_path = sales_path
        self.df_customers = None
        self.df_products = None
        self.df_sales = None
        self.df_merged = None
        self.rfm_data = None
        self.scaled_rfm = None
        self.kmeans_model = None
        self.cluster_labels = None

    def load_data(self):
        """Load and merge all datasets"""
        print("Loading datasets...")

        # Load datasets
        self.df_customers = pd.read_csv(self.customers_path)
        self.df_products = pd.read_csv(self.products_path)
        self.df_sales = pd.read_csv(self.sales_path)

        print(f"Customers dataset: {self.df_customers.shape}")
        print(f"Products dataset: {self.df_products.shape}")
        print(f"Sales dataset: {self.df_sales.shape}")

        # Convert invoice date to datetime
        self.df_sales['Invoice date'] = pd.to_datetime(self.df_sales['Invoice date'], format='%d/%m/%Y')

    def preprocess_data(self):
        """Preprocess and merge datasets"""
        print("\nPreprocessing data...")

        # Explode product lists in sales data
        self.df_sales['Product id list'] = self.df_sales['Product id list'].str.split(',')
        df_exploded = self.df_sales.explode('Product id list')
        df_exploded = df_exploded.rename(columns={'Product id list': 'Product id'})

        # Merge with products to get prices
        df_with_prices = df_exploded.merge(
            self.df_products[['Product id', 'Price']],
            on='Product id',
            how='left'
        )

        # Merge with customers
        self.df_merged = df_with_prices.merge(
            self.df_customers,
            on='Customer id',
            how='left'
        )

        print(f"Merged dataset: {self.df_merged.shape}")

    def calculate_rfm(self, snapshot_date=None):
        """
        Calculate RFM metrics
        Recency: Days since last purchase
        Frequency: Number of purchases
        Monetary: Total amount spent
        """
        print("\nCalculating RFM metrics...")

        if snapshot_date is None:
            snapshot_date = self.df_merged['Invoice date'].max()

        # Calculate RFM metrics
        rfm = self.df_merged.groupby('Customer id').agg({
            'Invoice date': lambda x: (snapshot_date - x.max()).days,  # Recency
            'Invoice no': 'nunique',  # Frequency
            'Price': 'sum'  # Monetary
        }).reset_index()

        rfm.columns = ['Customer id', 'Recency', 'Frequency', 'Monetary']

        self.rfm_data = rfm
        print(f"RFM data shape: {rfm.shape}")

        return rfm

    def explore_rfm_distribution(self):
        """Explore the distribution of RFM metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Recency distribution
        axes[0].hist(self.rfm_data['Recency'], bins=50, alpha=0.7, color='skyblue')
        axes[0].set_title('Recency Distribution')
        axes[0].set_xlabel('Days since last purchase')
        axes[0].set_ylabel('Number of Customers')

        # Frequency distribution
        axes[1].hist(self.rfm_data['Frequency'], bins=50, alpha=0.7, color='lightgreen')
        axes[1].set_title('Frequency Distribution')
        axes[1].set_xlabel('Number of purchases')
        axes[1].set_ylabel('Number of Customers')

        # Monetary distribution
        axes[2].hist(self.rfm_data['Monetary'], bins=50, alpha=0.7, color='salmon')
        axes[2].set_title('Monetary Distribution')
        axes[2].set_xlabel('Total amount spent')
        axes[2].set_ylabel('Number of Customers')

        plt.tight_layout()
        plt.show()

        # Print RFM statistics
        print("\nRFM Statistics:")
        print(self.rfm_data[['Recency', 'Frequency', 'Monetary']].describe())

    def prepare_data_for_clustering(self):
        """Prepare RFM data for clustering"""
        print("\nPreparing data for clustering...")

        # Handle outliers and skewness using log transformation
        rfm_log = self.rfm_data[['Recency', 'Frequency', 'Monetary']].copy()

        # For frequency and monetary, we use log transformation
        rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
        rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])

        # For recency, we might not need transformation as lower is better
        # But we'll standardize all features

        # Standardize the data
        scaler = StandardScaler()
        self.scaled_rfm = scaler.fit_transform(rfm_log)

        return self.scaled_rfm

    def find_optimal_clusters(self, max_k=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        print("\nFinding optimal number of clusters...")

        wcss = []  # Within-cluster sum of squares
        silhouette_scores = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_rfm)
            wcss.append(kmeans.inertia_)

            if k > 1:  # Silhouette score requires at least 2 clusters
                score = silhouette_score(self.scaled_rfm, kmeans.labels_)
                silhouette_scores.append(score)

        # Plot elbow curve and silhouette scores
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Elbow method
        axes[0].plot(k_range, wcss, 'bo-')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('WCSS')
        axes[0].set_title('Elbow Method')

        # Silhouette scores
        axes[1].plot(range(2, max_k + 1), silhouette_scores, 'ro-')
        axes[1].set_xlabel('Number of Clusters')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis')

        plt.tight_layout()
        plt.show()

        # Find optimal k (we'll use 4 as specified in requirements)
        optimal_k = 4
        print(f"Using {optimal_k} clusters as specified in requirements")

        return optimal_k

    def perform_kmeans_clustering(self, n_clusters=4):
        """Perform K-Means clustering"""
        print(f"\nPerforming K-Means clustering with {n_clusters} clusters...")

        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(self.scaled_rfm)

        # Add cluster labels to RFM data
        self.rfm_data['Cluster'] = self.cluster_labels

        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.scaled_rfm, self.cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")

        return self.cluster_labels

    def analyze_clusters(self):
        """Analyze and interpret the clusters"""
        print("\nAnalyzing clusters...")

        # Calculate cluster statistics
        cluster_stats = self.rfm_data.groupby('Cluster').agg({
            'Recency': ['mean', 'std', 'min', 'max'],
            'Frequency': ['mean', 'std', 'min', 'max'],
            'Monetary': ['mean', 'std', 'min', 'max'],
            'Customer id': 'count'
        }).round(2)

        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
        cluster_stats = cluster_stats.rename(columns={'Customer id_count': 'Customer_Count'})

        print("Cluster Statistics:")
        print(cluster_stats)

        # Visualize clusters
        self.visualize_clusters()

        return cluster_stats

    def visualize_clusters(self):
        """Visualize the clusters in 2D and 3D space"""

        # 2D Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Recency vs Frequency
        scatter1 = axes[0, 0].scatter(self.rfm_data['Recency'], self.rfm_data['Frequency'],
                                      c=self.rfm_data['Cluster'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel('Recency')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Recency vs Frequency')
        plt.colorbar(scatter1, ax=axes[0, 0])

        # Frequency vs Monetary
        scatter2 = axes[0, 1].scatter(self.rfm_data['Frequency'], self.rfm_data['Monetary'],
                                      c=self.rfm_data['Cluster'], cmap='viridis', alpha=0.6)
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Monetary')
        axes[0, 1].set_title('Frequency vs Monetary')
        plt.colorbar(scatter2, ax=axes[0, 1])

        # Recency vs Monetary
        scatter3 = axes[1, 0].scatter(self.rfm_data['Recency'], self.rfm_data['Monetary'],
                                      c=self.rfm_data['Cluster'], cmap='viridis', alpha=0.6)
        axes[1, 0].set_xlabel('Recency')
        axes[1, 0].set_ylabel('Monetary')
        axes[1, 0].set_title('Recency vs Monetary')
        plt.colorbar(scatter3, ax=axes[1, 0])

        # Customer distribution per cluster
        cluster_counts = self.rfm_data['Cluster'].value_counts().sort_index()
        axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color=['red', 'orange', 'green', 'blue'])
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Number of Customers')
        axes[1, 1].set_title('Customer Distribution per Cluster')

        plt.tight_layout()
        plt.show()

        # 3D Visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(self.rfm_data['Recency'],
                             self.rfm_data['Frequency'],
                             self.rfm_data['Monetary'],
                             c=self.rfm_data['Cluster'],
                             cmap='viridis', alpha=0.6)

        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        ax.set_title('3D RFM Clusters')
        plt.colorbar(scatter)

        plt.show()

    def assign_segment_labels(self):
        """Assign business segment labels based on cluster characteristics"""
        print("\nAssigning business segment labels...")

        # Calculate average RFM values for each cluster
        cluster_means = self.rfm_data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

        # Define segment mapping based on RFM characteristics
        segment_mapping = {}

        for cluster in cluster_means.index:
            recency = cluster_means.loc[cluster, 'Recency']
            frequency = cluster_means.loc[cluster, 'Frequency']
            monetary = cluster_means.loc[cluster, 'Monetary']

            # Business logic for segment assignment
            if monetary > cluster_means['Monetary'].quantile(0.75) and frequency > cluster_means['Frequency'].quantile(
                    0.75):
                segment = "High Value"
            elif recency > cluster_means['Recency'].quantile(0.75):
                segment = "At Risk"
            elif frequency > cluster_means['Frequency'].quantile(0.5) and monetary > cluster_means['Monetary'].quantile(
                    0.5):
                segment = "Regular"
            else:
                segment = "New/Low Activity"

            segment_mapping[cluster] = segment

        # Apply segment labels
        self.rfm_data['Segment'] = self.rfm_data['Cluster'].map(segment_mapping)

        print("Segment Distribution:")
        segment_counts = self.rfm_data['Segment'].value_counts()
        print(segment_counts)

        # Visualize segment distribution
        plt.figure(figsize=(10, 6))
        segment_counts.plot(kind='bar', color=['green', 'blue', 'orange', 'red'])
        plt.title('Customer Segment Distribution')
        plt.xlabel('Segment')
        plt.ylabel('Number of Customers')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return segment_mapping

    def generate_business_recommendations(self):
        """Generate business recommendations for each segment"""
        print("\nGenerating Business Recommendations...")

        segment_analysis = self.rfm_data.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Customer id': 'count'
        }).round(2)

        segment_analysis = segment_analysis.rename(columns={'Customer id': 'Customer_Count'})

        recommendations = {
            "High Value": {
                "Characteristics": "High spending, frequent purchases, recent activity",
                "Recommendations": [
                    "VIP treatment with exclusive offers",
                    "Personalized product recommendations",
                    "Loyalty program with premium benefits",
                    "Early access to new products",
                    "Dedicated customer service"
                ]
            },
            "Regular": {
                "Characteristics": "Moderate spending and frequency, stable activity",
                "Recommendations": [
                    "Cross-selling and up-selling opportunities",
                    "Moderate loyalty rewards",
                    "Seasonal promotions",
                    "Product bundle offers",
                    "Email marketing with personalized content"
                ]
            },
            "At Risk": {
                "Characteristics": "Long time since last purchase, potential churn",
                "Recommendations": [
                    "Win-back campaigns with special discounts",
                    "Re-engagement emails",
                    "Survey to understand reasons for inactivity",
                    "Limited-time exclusive offers",
                    "Personalized we miss you messages"
                ]
            },
            "New/Low Activity": {
                "Characteristics": "Low spending, infrequent purchases, or new customers",
                "Recommendations": [
                    "Welcome series for new customers",
                    "Educational content about products",
                    "Small incentive for second purchase",
                    "Simple loyalty program introduction",
                    "Social proof and testimonials"
                ]
            }
        }

        # Print recommendations
        for segment, info in recommendations.items():
            print(f"\n{'=' * 50}")
            print(f"SEGMENT: {segment}")
            print(f"{'=' * 50}")
            print(f"Characteristics: {info['Characteristics']}")
            print("Recommendations:")
            for i, rec in enumerate(info['Recommendations'], 1):
                print(f"  {i}. {rec}")

        return recommendations

    def create_segmentation_report(self):
        """Create comprehensive segmentation report"""
        print("\n" + "=" * 80)
        print("CUSTOMER SEGMENTATION & VALUE ANALYSIS REPORT")
        print("=" * 80)

        # Basic statistics
        total_customers = len(self.rfm_data)
        total_revenue = self.rfm_data['Monetary'].sum()

        print(f"\nOverall Statistics:")
        print(f"Total Customers: {total_customers:,}")
        print(f"Total Revenue: HKD {total_revenue:,.2f}")
        print(f"Average Revenue per Customer: HKD {total_revenue / total_customers:,.2f}")

        # Segment-wise analysis
        segment_summary = self.rfm_data.groupby('Segment').agg({
            'Customer id': 'count',
            'Monetary': ['sum', 'mean'],
            'Frequency': 'mean',
            'Recency': 'mean'
        }).round(2)

        segment_summary.columns = ['Customer_Count', 'Total_Revenue', 'Avg_Revenue', 'Avg_Frequency', 'Avg_Recency']
        segment_summary['Revenue_Share'] = (segment_summary['Total_Revenue'] / total_revenue * 100).round(2)
        segment_summary['Customer_Share'] = (segment_summary['Customer_Count'] / total_customers * 100).round(2)

        print(f"\nSegment Performance Summary:")
        print(segment_summary)

        # Visualization: Revenue vs Customer share
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Revenue share
        axes[0].pie(segment_summary['Revenue_Share'],
                    labels=segment_summary.index,
                    autopct='%1.1f%%',
                    colors=['green', 'blue', 'orange', 'red'])
        axes[0].set_title('Revenue Share by Segment')

        # Customer share
        axes[1].pie(segment_summary['Customer_Share'],
                    labels=segment_summary.index,
                    autopct='%1.1f%%',
                    colors=['green', 'blue', 'orange', 'red'])
        axes[1].set_title('Customer Share by Segment')

        plt.tight_layout()
        plt.show()

        return segment_summary

    def run_complete_analysis(self):
        """Run complete customer segmentation analysis"""
        print("Starting Customer Segmentation Analysis...")

        # Step 1: Load data
        self.load_data()

        # Step 2: Preprocess data
        self.preprocess_data()

        # Step 3: Calculate RFM
        self.calculate_rfm()

        # Step 4: Explore RFM distribution
        self.explore_rfm_distribution()

        # Step 5: Prepare for clustering
        self.prepare_data_for_clustering()

        # Step 6: Find optimal clusters (using 4 as specified)
        optimal_k = self.find_optimal_clusters()

        # Step 7: Perform clustering
        self.perform_kmeans_clustering(n_clusters=optimal_k)

        # Step 8: Analyze clusters
        self.analyze_clusters()

        # Step 9: Assign segment labels
        self.assign_segment_labels()

        # Step 10: Generate business recommendations
        self.generate_business_recommendations()

        # Step 11: Create final report
        final_report = self.create_segmentation_report()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        return self.rfm_data, final_report


# Execute the analysis
if __name__ == "__main__":
    # Get the directory where the current script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the correct file paths
    customers_path = 'group_2/customers_2.csv'
    products_path = 'group_2/products_2.csv'
    sales_path = 'group_2/sales_2.csv'
    
    # Check if files exist
    for path, name in [(customers_path, 'customers'), 
                       (products_path, 'products'), 
                       (sales_path, 'sales')]:
        if not os.path.exists(path):
            print(f"Error: {name} file not found at {path}")
    
    # Initialize the segmentation analysis
    segmentation = CustomerSegmentation(
        customers_path=customers_path,
        products_path=products_path,
        sales_path=sales_path
    )

    # Run complete analysis
    rfm_results, report = segmentation.run_complete_analysis()

    # Ensure output directory exists
    output_dir = os.path.join(current_dir, 'newCSV')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to CSV for further analysis
    output_path = os.path.join(output_dir, 's1_customer_segmentation_results.csv')
    rfm_results.to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")