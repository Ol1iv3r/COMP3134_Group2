import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the correct file path
file_path = 'group_2/customers_2.csv'

print(f"Attempting to load file: {file_path}")
print(f"File exists: {os.path.exists(file_path)}")

# Load data
df = pd.read_csv(file_path)


print("=== Basic Data Analysis ===")
print(f"Total number of customers: {len(df)}")
print(f"Gender distribution:\n{df['Gender'].value_counts()}")
print(f"\nPayment method distribution:\n{df['Payment method'].value_counts()}")
print(f"\nAge statistics:")
print(df['Age'].describe())


# Age group analysis
def age_group(age):
    if age < 30:
        return "18-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    else:
        return "60+"


df['Age_Group'] = df['Age'].apply(age_group)

# Visualization results
plt.figure(figsize=(15, 10))

# 1. Payment method distribution
plt.subplot(2, 2, 1)
payment_counts = df['Payment method'].value_counts()
plt.pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%')
plt.title('Payment Method Distribution')

# 2. Age distribution
plt.subplot(2, 2, 2)
age_groups = df['Age_Group'].value_counts().sort_index()
plt.bar(age_groups.index, age_groups.values)
plt.title('Age Group Distribution')
plt.xticks(rotation=45)
for i, v in enumerate(age_groups.values):
    plt.text(i, v, str(v), ha='center', va='bottom')

# 3. Payment method vs. Age group
plt.subplot(2, 2, 3)
cross_tab = pd.crosstab(df['Age_Group'], df['Payment method'])
cross_tab.plot(kind='bar', ax=plt.gca())
plt.title('Payment Method Preference by Age Group')
plt.xticks(rotation=45)
plt.legend(title='Payment Method')

# 4. Gender vs. Payment method
plt.subplot(2, 2, 4)
gender_payment = pd.crosstab(df['Gender'], df['Payment method'])
gender_payment.plot(kind='bar', ax=plt.gca())
plt.title('Payment Method Preference by Gender')
plt.legend(title='Payment Method')

plt.tight_layout()
plt.show()

# Key insights summary
print("\n" + "=" * 50)
print("Key Insights for Marketing Campaigns")
print("=" * 50)

# Payment preferences by age group
print("\nPayment Preferences by Age Group:")
age_payment = pd.crosstab(df['Age_Group'], df['Payment method'], normalize='index') * 100
print(age_payment.round(1))

# Generate marketing recommendations
print("\nMarketing Recommendations:")
for age_group in sorted(df['Age_Group'].unique()):
    group_data = df[df['Age_Group'] == age_group]
    preferred_payment = group_data['Payment method'].mode()[0]
    size = len(group_data)

    print(f"\n{age_group} age group ({size} people, {size / len(df) * 100:.1f}%):")
    print(f"  Most common payment method: {preferred_payment}")

    if age_group == "18-29":
        print("  Recommended strategy: Social media marketing, mobile payment promotions, youth-exclusive discounts")
    elif age_group == "30-39":
        print("  Recommended strategy: Family packages, credit card reward points, convenient payment experience")
    elif age_group == "40-49":
        print("  Recommended strategy: Quality assurance, membership privileges, diverse payment options")
    elif age_group == "50-59":
        print("  Recommended strategy: Traditional channel promotion, large discounts, credit payment promotion")
    else:
        print("  Recommended strategy: Simple and easy-to-use payment methods, offline events, attentive service")

# In-depth payment method analysis
print("\n" + "=" * 50)
print("In-depth Payment Method Analysis")
print("=" * 50)
for payment in df['Payment method'].unique():
    payment_data = df[df['Payment method'] == payment]
    avg_age = payment_data['Age'].mean()
    gender_ratio = payment_data['Gender'].value_counts(normalize=True)['F'] * 100

    print(f"\n{payment} users:")
    print(f"  Average age: {avg_age:.1f} years old")
    print(f"  Female ratio: {gender_ratio:.1f}%")
    print(f"  Number of users: {len(payment_data)} people")