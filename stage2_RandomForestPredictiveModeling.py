import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CustomerLTVPredictor:
    def __init__(self, customers_path, products_path, sales_path, segmentation_path):
        """
        Initialize the Customer LTV Prediction analysis
        """
        self.customers_path = customers_path
        self.products_path = products_path
        self.sales_path = sales_path
        self.segmentation_path = segmentation_path
        self.df_customers = None
        self.df_products = None
        self.df_sales = None
        self.df_segmentation = None
        self.df_merged = None
        self.feature_importance = None
        self.rf_regressor = None
        self.rf_classifier = None
        self.preprocessor = None
        self.label_encoder = None

    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")

        # Check if files exist
        required_files = [
            (self.customers_path, 'customers'),
            (self.products_path, 'products'), 
            (self.sales_path, 'sales'),
            (self.segmentation_path, 'segmentation')
        ]
        
        for file_path, file_name in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_name} file not found: {file_path}")

        # Load datasets
        self.df_customers = pd.read_csv(self.customers_path)
        self.df_products = pd.read_csv(self.products_path)
        self.df_sales = pd.read_csv(self.sales_path)
        self.df_segmentation = pd.read_csv(self.segmentation_path)

        print(f"Customers dataset: {self.df_customers.shape}")
        print(f"Products dataset: {self.df_products.shape}")
        print(f"Sales dataset: {self.df_sales.shape}")
        print(f"Segmentation dataset: {self.df_segmentation.shape}")

        # Convert invoice date to datetime
        self.df_sales['Invoice date'] = pd.to_datetime(self.df_sales['Invoice date'], format='%d/%m/%Y')

    def preprocess_and_engineer_features(self):
        """Preprocess data and engineer features for prediction"""
        print("\nPreprocessing data and engineering features...")

        # Explode product lists in sales data
        self.df_sales['Product id list'] = self.df_sales['Product id list'].str.split(',')
        df_exploded = self.df_sales.explode('Product id list')
        df_exploded = df_exploded.rename(columns={'Product id list': 'Product id'})

        # Merge with products to get prices and categories
        df_with_products = df_exploded.merge(
            self.df_products[['Product id', 'Price', 'Category']],
            on='Product id',
            how='left'
        )

        # Merge with customers
        self.df_merged = df_with_products.merge(
            self.df_customers,
            on='Customer id',
            how='left'
        )

        # Merge with segmentation results
        self.df_merged = self.df_merged.merge(
            self.df_segmentation[['Customer id', 'Segment']],
            on='Customer id',
            how='left'
        )

        print(f"Merged dataset shape: {self.df_merged.shape}")

    def create_ltv_features(self):
        """Create features for LTV prediction"""
        print("\nCreating LTV prediction features...")

        # Calculate snapshot date (most recent date in data)
        snapshot_date = self.df_merged['Invoice date'].max()

        # Feature engineering for each customer
        customer_features = []

        for customer_id in self.df_merged['Customer id'].unique():
            customer_data = self.df_merged[self.df_merged['Customer id'] == customer_id]

            # Basic RFM features
            recency = (snapshot_date - customer_data['Invoice date'].max()).days
            frequency = customer_data['Invoice no'].nunique()
            monetary = customer_data['Price'].sum()
            avg_order_value = monetary / frequency if frequency > 0 else 0

            # Behavioral features
            tenure_days = (customer_data['Invoice date'].max() - customer_data['Invoice date'].min()).days
            avg_days_between_orders = tenure_days / (frequency - 1) if frequency > 1 else 0

            # Product behavior features
            unique_products = customer_data['Product id'].nunique()
            product_categories = customer_data['Category'].nunique()
            avg_products_per_order = len(customer_data) / frequency

            # Mall behavior features
            mall_visits = customer_data['Shopping mall'].value_counts()
            unique_malls = customer_data['Shopping mall'].nunique()
            favorite_mall = mall_visits.index[0] if len(mall_visits) > 0 else 'Unknown'

            # Time-based features
            monthly_purchase_pattern = customer_data.groupby(
                customer_data['Invoice date'].dt.to_period('M')
            )['Price'].sum()
            purchase_consistency = monthly_purchase_pattern.std() if len(monthly_purchase_pattern) > 1 else 0

            # Customer demographic features
            customer_info = self.df_customers[self.df_customers['Customer id'] == customer_id].iloc[0]
            age = customer_info['Age']
            gender = customer_info['Gender']
            payment_method = customer_info['Payment method']

            # Segment from previous analysis
            segment = customer_data['Segment'].iloc[0] if 'Segment' in customer_data.columns else 'Unknown'

            customer_features.append({
                'Customer id': customer_id,
                'Recency': recency,
                'Frequency': frequency,
                'Monetary': monetary,
                'Avg_Order_Value': avg_order_value,
                'Tenure_Days': tenure_days,
                'Avg_Days_Between_Orders': avg_days_between_orders,
                'Unique_Products': unique_products,
                'Product_Categories': product_categories,
                'Avg_Products_Per_Order': avg_products_per_order,
                'Unique_Malls': unique_malls,
                'Purchase_Consistency': purchase_consistency,
                'Age': age,
                'Gender': gender,
                'Payment_Method': payment_method,
                'Segment': segment
            })

        features_df = pd.DataFrame(customer_features)
        return features_df

    def prepare_ltv_target(self, features_df, future_months=6):
        """Prepare target variable for LTV prediction"""
        print(f"\nPreparing LTV target variable for next {future_months} months...")

        # Calculate future value based on historical patterns
        snapshot_date = self.df_merged['Invoice date'].max()
        future_cutoff = snapshot_date - pd.DateOffset(months=future_months)

        # Use recent months as proxy for future value
        recent_data = self.df_merged[self.df_merged['Invoice date'] > future_cutoff]
        future_ltv = recent_data.groupby('Customer id')['Price'].sum().reset_index()
        future_ltv.columns = ['Customer id', 'Future_LTV']

        # Merge with features
        features_with_target = features_df.merge(future_ltv, on='Customer id', how='left')
        features_with_target['Future_LTV'] = features_with_target['Future_LTV'].fillna(0)

        # Create classification target (High/Low Value)
        ltv_median = features_with_target['Future_LTV'].median()
        features_with_target['LTV_Class'] = features_with_target['Future_LTV'].apply(
            lambda x: 'High' if x > ltv_median else 'Low'
        )

        return features_with_target

    def create_preprocessor(self, features_df):
        """Create preprocessing pipeline for categorical and numerical features"""
        print("\nCreating preprocessing pipeline...")

        # Separate features
        categorical_features = ['Gender', 'Payment_Method', 'Segment']
        numerical_features = [col for col in features_df.columns
                              if col not in categorical_features + ['Customer id', 'Future_LTV', 'LTV_Class']
                              and features_df[col].dtype in ['int64', 'float64']]

        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ])

        return numerical_features, categorical_features

    def train_random_forest_models(self, features_with_target):
        """Train Random Forest models for LTV prediction"""
        print("\nTraining Random Forest models...")

        # Create preprocessor
        numerical_features, categorical_features = self.create_preprocessor(features_with_target)

        # Prepare features and targets
        feature_columns = numerical_features + categorical_features
        X = features_with_target[feature_columns]
        y_regression = features_with_target['Future_LTV']

        # Encode classification target
        self.label_encoder = LabelEncoder()
        y_classification = self.label_encoder.fit_transform(features_with_target['LTV_Class'])

        # Split data
        X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
            X, y_regression, y_classification, test_size=0.2, random_state=42
        )

        # Preprocess the training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # Train Regression Model (Predict LTV value)
        print("Training Random Forest Regressor...")
        self.rf_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_regressor.fit(X_train_processed, y_reg_train)

        # Train Classification Model (Predict High/Low Value)
        print("Training Random Forest Classifier...")
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_classifier.fit(X_train_processed, y_cls_train)

        # Evaluate models
        self.evaluate_models(X_test_processed, y_reg_test, y_cls_test)

        return X, feature_columns

    def evaluate_models(self, X_test_processed, y_reg_test, y_cls_test):
        """Evaluate model performance"""
        print("\nEvaluating model performance...")

        # Regression evaluation
        y_reg_pred = self.rf_regressor.predict(X_test_processed)
        rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
        r2 = r2_score(y_reg_test, y_reg_pred)

        print(f"Regression Model Performance:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")

        # Classification evaluation
        y_cls_pred = self.rf_classifier.predict(X_test_processed)

        # Convert encoded labels back to original
        y_cls_test_original = self.label_encoder.inverse_transform(y_cls_test)
        y_cls_pred_original = self.label_encoder.inverse_transform(y_cls_pred)

        print(f"\nClassification Model Performance:")
        print(classification_report(y_cls_test_original, y_cls_pred_original))

        # Feature importance
        print("\nCalculating feature importance...")

        # Get feature names
        numerical_features = self.preprocessor.named_transformers_['num'].feature_names_in_
        categorical_features = self.preprocessor.named_transformers_['cat'].feature_names_in_

        # Get OneHot encoded feature names
        ohe_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

        # Combine all feature names
        all_feature_names = list(numerical_features) + list(ohe_feature_names)

        # Ensure feature names and importance array lengths match
        if len(all_feature_names) == len(self.rf_regressor.feature_importances_):
            self.feature_importance = pd.DataFrame({
                'feature': all_feature_names,
                'importance': self.rf_regressor.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10))
        else:
            print(
                f"Warning: Feature names length ({len(all_feature_names)}) doesn't match importance length ({len(self.rf_regressor.feature_importances_)})")
            # Create a simple feature importance table
            self.feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(self.rf_regressor.feature_importances_))],
                'importance': self.rf_regressor.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nTop 10 Most Important Features (generic names):")
            print(self.feature_importance.head(10))

    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        print("\nAnalyzing feature importance...")

        if self.feature_importance is None:
            print("No feature importance data available.")
            return None

        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(15)

        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Features for LTV Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        return self.feature_importance

    def predict_future_trends(self, features_df, feature_columns):
        """Predict future trends and customer value"""
        print("\nPredicting future trends...")

        # Prepare features for prediction
        X = features_df[feature_columns]
        X_processed = self.preprocessor.transform(X)

        # Make predictions for all customers
        ltv_predictions = self.rf_regressor.predict(X_processed)
        value_class_predictions_encoded = self.rf_classifier.predict(X_processed)

        # Convert encoded predictions back to original labels
        value_class_predictions = self.label_encoder.inverse_transform(value_class_predictions_encoded)

        # Create prediction results
        predictions_df = features_df[['Customer id']].copy()
        predictions_df['Predicted_LTV'] = ltv_predictions
        predictions_df['Predicted_Value_Class'] = value_class_predictions

        # Add actual values for comparison
        if 'Future_LTV' in features_df.columns:
            predictions_df['Actual_LTV'] = features_df['Future_LTV']
            predictions_df['Actual_Value_Class'] = features_df['LTV_Class']

        return predictions_df

    def analyze_prediction_results(self, predictions_df):
        """Analyze prediction results and generate insights"""
        print("\nAnalyzing prediction results...")

        # Summary statistics
        print("Prediction Results Summary:")
        print(f"Total Customers: {len(predictions_df)}")
        print(
            f"Predicted High Value Customers: {len(predictions_df[predictions_df['Predicted_Value_Class'] == 'High'])}")
        print(f"Predicted Low Value Customers: {len(predictions_df[predictions_df['Predicted_Value_Class'] == 'Low'])}")
        print(f"Average Predicted LTV: {predictions_df['Predicted_LTV'].mean():.2f}")

        # Visualize predictions - fix chart display issues
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Predicted LTV distribution
        axes[0, 0].hist(predictions_df['Predicted_LTV'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribution of Predicted LTV')
        axes[0, 0].set_xlabel('Predicted LTV')
        axes[0, 0].set_ylabel('Number of Customers')

        # Value class distribution - fix pie chart display
        value_counts = predictions_df['Predicted_Value_Class'].value_counts()
        colors = ['lightgreen', 'lightcoral']  # Ensure correct color order

        # Ensure consistent label order
        labels = ['High', 'Low']
        sizes = [value_counts.get('High', 0), value_counts.get('Low', 0)]

        wedges, texts, autotexts = axes[0, 1].pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        axes[0, 1].set_title('Predicted Customer Value Distribution')

        # Enhance pie chart text readability
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')

        # Feature importance (if available)
        if self.feature_importance is not None:
            top_10 = self.feature_importance.head(10)
            axes[1, 0].barh(top_10['feature'], top_10['importance'])
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].set_xlabel('Importance Score')
        else:
            axes[1, 0].text(0.5, 0.5, 'Feature Importance\nNot Available',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Importance')

        # Actual vs Predicted (if available)
        if 'Actual_LTV' in predictions_df.columns:
            axes[1, 1].scatter(predictions_df['Actual_LTV'], predictions_df['Predicted_LTV'], alpha=0.6)
            max_val = max(predictions_df['Actual_LTV'].max(), predictions_df['Predicted_LTV'].max())
            axes[1, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
            axes[1, 1].set_xlabel('Actual LTV')
            axes[1, 1].set_ylabel('Predicted LTV')
            axes[1, 1].set_title('Actual vs Predicted LTV')
        else:
            # If no actual values, display box plot of predicted values
            boxplot_data = []
            boxplot_labels = []

            for value_class in ['High', 'Low']:
                class_data = predictions_df[predictions_df['Predicted_Value_Class'] == value_class]['Predicted_LTV']
                if len(class_data) > 0:
                    boxplot_data.append(class_data)
                    boxplot_labels.append(value_class)

            if boxplot_data:
                axes[1, 1].boxplot(boxplot_data, labels=boxplot_labels)
                axes[1, 1].set_ylabel('Predicted LTV')
                axes[1, 1].set_title('LTV Distribution by Value Class')
            else:
                axes[1, 1].text(0.5, 0.5, 'No data available\nfor boxplot',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('LTV Distribution')

        plt.tight_layout()
        plt.show()

        # Display enhanced value distribution pie chart separately for clarity
        self.display_enhanced_value_distribution(predictions_df)

        return predictions_df

    def display_enhanced_value_distribution(self, predictions_df):
        """Display enhanced value distribution chart separately"""
        print("\nDisplaying enhanced value distribution analysis...")

        # Create more detailed distribution analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Pie chart - more detailed settings
        value_counts = predictions_df['Predicted_Value_Class'].value_counts()
        colors = ['#2E8B57', '#CD5C5C']  # More vivid colors

        # Ensure correct label order
        labels = []
        sizes = []
        for label in ['High', 'Low']:
            if label in value_counts.index:
                labels.append(label)
                sizes.append(value_counts[label])

        wedges, texts, autotexts = axes[0].pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.05, 0.05] if len(sizes) == 2 else [0.05],  # Slightly separate wedges
            shadow=True
        )

        axes[0].set_title('Customer Value Distribution\n(Predicted)', fontsize=14, fontweight='bold')

        # Enhance text style
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')

        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')

        # Bar chart showing specific counts
        axes[1].bar(value_counts.index, value_counts.values, color=colors, alpha=0.7)
        axes[1].set_title('Customer Count by Value Class', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Number of Customers')
        axes[1].set_xlabel('Value Class')

        # Display values on bar chart
        for i, v in enumerate(value_counts.values):
            axes[1].text(i, v + max(value_counts.values) * 0.01, str(v),
                         ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

        # Print detailed statistics
        print(f"\n📊 Detailed Value Distribution:")
        for value_class in ['High', 'Low']:
            count = value_counts.get(value_class, 0)
            percentage = (count / len(predictions_df)) * 100
            avg_ltv = predictions_df[predictions_df['Predicted_Value_Class'] == value_class]['Predicted_LTV'].mean()
            print(f"• {value_class} Value: {count} customers ({percentage:.1f}%), Average LTV: {avg_ltv:.2f}")

    def generate_business_recommendations(self, predictions_df, feature_importance):
        """Generate business recommendations based on predictions"""
        print("\nGenerating Business Recommendations...")

        # Key insights from predictions
        high_value_customers = predictions_df[predictions_df['Predicted_Value_Class'] == 'High']
        low_value_customers = predictions_df[predictions_df['Predicted_Value_Class'] == 'Low']

        total_customers = len(predictions_df)
        high_value_ratio = len(high_value_customers) / total_customers

        print("=" * 60)
        print("BUSINESS INTELLIGENCE REPORT")
        print("=" * 60)

        print(f"\n📊 PREDICTION INSIGHTS:")
        print(f"• {len(high_value_customers)} customers predicted as High Value ({high_value_ratio * 100:.1f}%)")
        print(f"• {len(low_value_customers)} customers predicted as Low Value ({(1 - high_value_ratio) * 100:.1f}%)")
        print(f"• Average Predicted LTV: HKD {predictions_df['Predicted_LTV'].mean():,.2f}")

        if feature_importance is not None:
            print(f"\n🎯 KEY DRIVERS OF CUSTOMER VALUE (Top 5):")
            for i, row in feature_importance.head(5).iterrows():
                print(f"  {i + 1}. {row['feature']} (importance: {row['importance']:.3f})")

        print(f"\n💡 STRATEGIC RECOMMENDATIONS:")

        recommendations = {
            "Resource Allocation": [
                f"Focus 70% of marketing budget on {len(high_value_customers)} high-value customers",
                "Implement tiered service levels based on predicted value",
                "Develop personalized retention programs for at-risk high-value customers"
            ],
            "Customer Acquisition": [
                "Use feature importance to identify characteristics of high-value customers",
                "Optimize acquisition channels based on predicted customer value",
                "Develop targeted campaigns for demographics with higher predicted LTV"
            ],
            "Retention Strategy": [
                "Create early warning system for customers likely to decrease in value",
                "Implement proactive engagement for high-value customers showing risk signals",
                "Develop win-back campaigns based on predicted value potential"
            ],
            "Product Development": [
                "Analyze product preferences of high-value customer segments",
                "Develop bundled offerings based on purchasing patterns",
                "Create premium services for predicted high-value segments"
            ]
        }

        for category, recs in recommendations.items():
            print(f"\n{category}:")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec}")

        print(f"\n📈 GROWTH OPPORTUNITIES:")
        growth_opportunities = [
            f"Upsell potential: {len(low_value_customers[low_value_customers['Predicted_LTV'] > 0])} low-value customers have positive predicted LTV",
            f"Cross-sell opportunity: Focus on increasing product categories per customer",
            "Seasonal optimization: Align marketing with high-value customer purchase patterns",
            "Channel optimization: Invest in channels that attract high-value customers"
        ]

        for opportunity in growth_opportunities:
            print(f"• {opportunity}")

        return recommendations

    def forecast_future_trends(self, features_df, feature_columns, months=12):
        """Forecast future business trends"""
        print(f"\nForecasting trends for next {months} months...")

        # Use the current high-value ratio as baseline
        high_value_customers = features_df[features_df['LTV_Class'] == 'High']
        current_high_value_ratio = len(high_value_customers) / len(features_df) if len(features_df) > 0 else 0

        # Simple growth projection based on historical patterns
        monthly_growth_rate = 0.02  # 2% monthly growth assumption

        print("📊 FUTURE TREND FORECAST:")
        print(f"Current High-Value Customer Ratio: {current_high_value_ratio * 100:.1f}%")

        for month in range(3, months + 1, 3):
            projected_ratio = current_high_value_ratio * (1 + monthly_growth_rate) ** month
            projected_revenue_growth = (projected_ratio - current_high_value_ratio) * 100

            print(f"Month {month}:")
            print(f"  • Projected High-Value Ratio: {projected_ratio * 100:.1f}%")
            print(f"  • Expected Revenue Growth: +{projected_revenue_growth:.1f}%")

        # Risk assessment
        at_risk_customers = features_df[
            (features_df['Recency'] > 180) &
            (features_df['Predicted_LTV'] > features_df['Predicted_LTV'].quantile(0.75))
            ] if 'Predicted_LTV' in features_df.columns else pd.DataFrame()

        print(f"\n⚠️  RISK ASSESSMENT:")
        print(f"• {len(at_risk_customers)} high-value customers at risk of churn (Recency > 180 days)")
        if len(at_risk_customers) > 0:
            print(f"• Potential revenue at risk: HKD {at_risk_customers['Predicted_LTV'].sum():,.2f}")

    def run_complete_analysis(self):
        """Run complete LTV prediction analysis"""
        print("Starting Customer LTV Prediction Analysis...")
        print("=" * 60)

        try:
            # Step 1: Load data
            self.load_data()

            # Step 2: Preprocess and engineer features
            self.preprocess_and_engineer_features()

            # Step 3: Create LTV features
            features_df = self.create_ltv_features()

            # Step 4: Prepare target variable
            features_with_target = self.prepare_ltv_target(features_df)

            # Step 5: Train Random Forest models
            X, feature_columns = self.train_random_forest_models(features_with_target)

            # Step 6: Analyze feature importance
            feature_importance = self.analyze_feature_importance()

            # Step 7: Make predictions
            predictions_df = self.predict_future_trends(features_with_target, feature_columns)

            # Step 8: Analyze results
            self.analyze_prediction_results(predictions_df)

            # Step 9: Generate business recommendations
            recommendations = self.generate_business_recommendations(predictions_df, feature_importance)

            # Step 10: Forecast future trends
            self.forecast_future_trends(features_with_target, feature_columns)

            print("\n" + "=" * 60)
            print("LTV PREDICTION ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)

            # Ensure output directory exists
            os.makedirs('./newCSV', exist_ok=True)

            # Save results to CSV for further analysis
            predictions_df.to_csv('./newCSV/s2_customer_ltv_predictions.csv', index=False)
            if feature_importance is not None:
                feature_importance.to_csv('./newCSV/s2_feature_importance_analysis.csv', index=False)

            print("\nResults saved to:")
            print("• s2_customer_ltv_predictions.csv")
            print("• s2_feature_importance_analysis.csv")

            return predictions_df, feature_importance, recommendations

        except Exception as e:
            print(f"\n❌❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            print("Attempting to continue with available results...")
            return None, None, None

# Execute the analysis
if __name__ == "__main__":
    # Construct the correct file paths (relative to the script's execution directory)
    customers_path = 'group_2/customers_2.csv'
    products_path = 'group_2/products_2.csv'
    sales_path = 'group_2/sales_2.csv'
    segmentation_path = 'newCSV/s1_customer_segmentation_results.csv'
    
    print("Checking file paths:")
    print(f"Customers: {customers_path} - Exists: {os.path.exists(customers_path)}")
    print(f"Products: {products_path} - Exists: {os.path.exists(products_path)}")
    print(f"Sales: {sales_path} - Exists: {os.path.exists(sales_path)}")
    print(f"Segmentation: {segmentation_path} - Exists: {os.path.exists(segmentation_path)}")
    
    # Check if Stage 1 results exist
    if not os.path.exists(segmentation_path):
        print("\n⚠️ Warning: Stage 1 analysis results file not found")
        print("Please run stage1_RFM_Kmeans.py first to generate customer segmentation results")
        print("Or use the following command to create an example file for testing:")
        print("mkdir -p newCSV")
        print("# Code to create example segmentation file...")
        
        # Option to create example file or exit
        response = input("Do you want to create an example segmentation file to continue testing? (y/n): ")
        if response.lower() == 'y':
            # Create example segmentation file
            os.makedirs('./newCSV', exist_ok=True)
            # Add code to create example file here
            print("Example file creation functionality not implemented, please run Stage 1 analysis first")
            exit(1)
        else:
            exit(1)

    # Initialize the LTV prediction analysis
    ltv_predictor = CustomerLTVPredictor(
        customers_path=customers_path,
        products_path=products_path,
        sales_path=sales_path,
        segmentation_path=segmentation_path
    )

    # Run complete analysis
    predictions, feature_importance, recommendations = ltv_predictor.run_complete_analysis()