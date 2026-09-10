<div align="center">

# 🛒 SmartCart Customer Segmentation.

### Discovering Hidden Customer Patterns with Unsupervised Machine Learning

<p>
  <img src="https://img.shields.io/badge/Machine%20Learning-Unsupervised-6C63FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/K--Means-Clustering-FF6B6B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Agglomerative-Clustering-00B894?style=for-the-badge" />
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" />
</p>

<br>

> **An end-to-end machine learning project that transforms raw customer data into meaningful customer segments using clustering algorithms.**

<br>

**👨‍💻 Created by Sumit Jha**  
*Data Science • Machine Learning • Turning Data into Insights*

</div>

---

## ✨ Overview

Modern businesses collect large amounts of customer data, but raw data alone does not provide actionable insights.

The challenge is understanding:

> **Who are the customers, how do they behave, what are their spending patterns, and can meaningful groups be discovered without predefined labels?**

This project uses **Unsupervised Machine Learning** to identify hidden patterns within SmartCart's customer data.

Instead of manually defining customer categories, clustering algorithms allow the **data itself to reveal natural customer groups**.

The project analyzes customer:

- 👤 Demographics
- 💰 Income
- 🛒 Product spending
- 🌐 Online purchasing behavior
- 🏪 Store purchasing behavior
- 🎟️ Discount purchases
- 📊 Purchase frequency
- ⏳ Customer recency
- 📅 Customer relationship information

---

# 🎯 Business Problem

Treating every customer the same can lead to inefficient marketing and poor personalization.

Different customers may:

<table>
<tr>
<td width="50%">

### 💎 High-Value Customers

Customers with higher spending and purchasing activity.

</td>

<td width="50%">

### 🎟️ Deal-Oriented Customers

Customers who frequently respond to discounts and promotions.

</td>
</tr>

<tr>
<td width="50%">

### 🌐 Digital Customers

Customers who prefer online purchasing channels.

</td>

<td width="50%">

### 🏪 Traditional Shoppers

Customers who primarily interact through physical stores.

</td>
</tr>
</table>

The goal is to discover these behavioral patterns and create a foundation for:

🎯 **Personalized Marketing** • 💎 **Customer Value Analysis** • 📈 **Retention Strategies** • 🛒 **Better Customer Understanding**

---

# 📊 Dataset

The SmartCart dataset contains customer information across demographics and purchasing behavior.

<div align="center">

| 👥 Customers | 📊 Attributes | 🤖 Learning Type |
|:---:|:---:|:---:|
| **2,240** | **22** | **Unsupervised Learning** |

</div>

### Key Data Categories

```text
                    SMARTCART DATASET
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   👤 Demographics    🛒 Spending       📊 Purchases
        │                  │                  │
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
                    🤖 MACHINE LEARNING
```

The dataset includes information such as:

- Customer demographics
- Household income
- Education and marital status
- Product category spending
- Website purchases
- Catalog purchases
- Store purchases
- Discount purchases
- Website visits
- Customer recency
- Customer feedback

---

# 🧠 Machine Learning Pipeline

The project follows a complete data science workflow.

```text
┌─────────────────────────────┐
│     RAW CUSTOMER DATA       │
│                             │
│   2,240 Customers           │
│   22 Attributes             │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   DATA PREPROCESSING        │
│                             │
│ • Missing Values            │
│ • Data Cleaning             │
│ • Data Transformation       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   FEATURE ENGINEERING       │
│                             │
│ • Age                       │
│ • Customer Tenure           │
│ • Total Spending            │
│ • Household Features        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ EXPLORATORY DATA ANALYSIS   │
│                             │
│ • Distributions             │
│ • Relationships             │
│ • Correlations              │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ FEATURE TRANSFORMATION      │
│                             │
│ • Encoding                  │
│ • Standard Scaling          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ DIMENSIONALITY REDUCTION    │
│                             │
│            PCA              │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ OPTIMAL CLUSTER SELECTION   │
│                             │
│ • Elbow Method              │
│ • Knee Detection            │
└──────────────┬──────────────┘
               │
               ▼
        ┌──────┴──────┐
        ▼             ▼
    K-MEANS     AGGLOMERATIVE
        │             │
        └──────┬──────┘
               ▼
      CUSTOMER SEGMENTS
```

---

# 🧹 Data Preprocessing

Real-world data is rarely ready for machine learning.

The preprocessing stage prepares the customer dataset for analysis.

### Key Steps

- 🔍 Dataset inspection
- ❓ Missing value analysis
- 🧹 Data cleaning
- 🔄 Data transformation
- 🔢 Numerical feature preparation

### Missing Values

The dataset contains missing values in customer income information.

Appropriate preprocessing techniques are applied to ensure the dataset remains suitable for clustering.

```text
RAW DATA
   │
   ▼
Missing Value Detection
   │
   ▼
Data Treatment
   │
   ▼
CLEAN DATASET
```

---

# 🧪 Feature Engineering

Raw data becomes more useful when meaningful features are derived from existing information.

### 🎂 Customer Age

```text
Year_Birth
     │
     ▼
Current Year - Year_Birth
     │
     ▼
Customer Age
```

### ⏳ Customer Tenure

Customer enrollment information is used to understand the length of the customer relationship.

```text
Dt_Customer
     │
     ▼
Enrollment Information
     │
     ▼
Customer Tenure
```

### 💰 Total Spending

Product-level spending can be combined to provide a broader representation of customer value.

```text
Wine     Fruits     Meat
  │        │         │
  └────────┼─────────┘
           │
           ▼
     TOTAL SPENDING
           ▲
           │
  ┌────────┼─────────┐
  │        │         │
Fish     Sweets      Gold
```

---

# 🔍 Exploratory Data Analysis

Before applying machine learning, the dataset is explored to understand customer behavior.

The analysis focuses on:

📊 **Feature distributions**  
🔥 **Correlation analysis**  
📈 **Customer spending patterns**  
🔗 **Relationships between variables**  
🔎 **Outlier inspection**

This stage helps reveal important relationships before clustering.

---

# 📏 Feature Scaling

Customer features exist on very different numerical scales.

For example:

```text
Income             → Large numerical values
Total Spending     → Monetary values
Web Purchases      → Small integers
Recency            → Days
Web Visits         → Small integers
```

Since clustering algorithms rely on distances between observations, features are standardized using:

# `StandardScaler`

```text
RAW FEATURES
      │
      ▼
┌───────────────┐
│ StandardScaler│
└───────┬───────┘
        │
        ▼
 SCALED FEATURES
        │
        ▼
FAIR FEATURE COMPARISON
```

This prevents large-scale variables from dominating the clustering process.

---

# 🧬 Dimensionality Reduction

The project uses **Principal Component Analysis (PCA)** to reduce the complexity of the feature space.

PCA helps transform high-dimensional customer information into a smaller number of meaningful components.

```text
High-Dimensional Data

Income
Age
Spending
Purchases
Web Activity
Recency
      │
      ▼
     PCA
      │
      ▼
┌───────────────────┐
│ Principal Comp. 1 │
│ Principal Comp. 2 │
│ Principal Comp. 3 │
└───────────────────┘
```

This makes the clustering structure easier to analyze and visualize.

---

# 🎯 Finding the Optimal Number of Clusters

One of the most important clustering decisions is determining:

> **How many customer segments should be created?**

The project evaluates different values of `K`.

### 📉 Elbow Method

The Elbow Method helps identify the point where adding additional clusters provides diminishing improvements.

```text
WCSS
 │
 │ ●
 │  ●
 │   ●
 │    ●
 │      ●
 │       ●
 └──────────────────────► K
             ▲
           ELBOW
```

### 🦵 Knee Detection

Automated knee detection is also used to support the identification of an appropriate cluster count.

This reduces dependence on purely subjective visual interpretation.

---

# 🤖 Clustering Algorithms

## 1️⃣ K-Means Clustering

K-Means groups customers based on similarity.

The algorithm attempts to minimize the distance between customers and their respective cluster centroids.

```text
      CUSTOMER SPACE

   ● ● ● ●       ▲ Cluster A
  ● ● ● ●
   ● ● ●


                  ● ● ●
                ● ● ● ●       ▲ Cluster B
                  ● ●


   ● ● ●
 ● ● ● ●             ▲ Cluster C
   ● ● ●
```

### Why K-Means?

- Efficient
- Popular and widely used
- Easy to interpret
- Suitable for numerical data
- Effective for segmentation problems

---

## 2️⃣ Agglomerative Clustering

Agglomerative Clustering uses a hierarchical approach.

It progressively combines similar observations into larger groups.

```text
Customers

 ●   ●   ●   ●   ●

 │   │   │   │   │

 └───┘   │   └───┘
    │    │     │
    └────┘     │
       │       │
       └───┬───┘
           │
           ▼

     CUSTOMER GROUPS
```

Using multiple clustering approaches provides different perspectives on the structure of the customer population.

---

# 📊 Cluster Evaluation

Because clustering does not use predefined labels, cluster quality must be evaluated differently from classification models.

The project explores clustering using:

<div align="center">

| 📉 | 🦵 | 📊 | 🎨 |
|---|---|---|---|
| Elbow Method | Knee Detection | Silhouette Analysis | Visual Inspection |

</div>

The objective is to create groups that demonstrate:

```text
High Similarity
WITHIN a Cluster

        +

Clear Separation
BETWEEN Clusters
```

---

# 🎨 Cluster Visualization

PCA-based dimensionality reduction allows customers to be visualized in a simplified feature space.

```text
                 PCA Component 2
                       ▲

            🔵 🔵
         🔵 🔵 🔵

                         🟢 🟢
                       🟢 🟢 🟢


     🔴 🔴
   🔴 🔴 🔴

────────────────────────────────► PCA Component 1
```

Visualization helps explore:

- Cluster separation
- Customer density
- Similar customer groups
- Potential overlap between clusters

---

# 💼 Business Value

Customer segmentation can support smarter and more personalized business decisions.

<table>
<tr>
<td width="50%">

## 🎯 Personalized Marketing

Different customer groups can potentially receive more relevant campaigns.

</td>

<td width="50%">

## 💎 Customer Value Analysis

Spending patterns can help identify groups with different purchasing behavior.

</td>
</tr>

<tr>
<td width="50%">

## 🌐 Channel Analysis

Clusters can help understand preferences for web, catalog, and store purchases.

</td>

<td width="50%">

## 📈 Customer Engagement

Recency and purchasing behavior can support retention strategies.

</td>
</tr>
</table>

---

# 🛠️ Technology Stack

<div align="center">

| Technology | Purpose |
|:---|:---|
| 🐍 Python | Core Programming Language |
| 🐼 Pandas | Data Manipulation |
| 🔢 NumPy | Numerical Computing |
| 📊 Matplotlib | Data Visualization |
| 🎨 Seaborn | Statistical Visualization |
| 🤖 Scikit-learn | Machine Learning |
| 🧬 PCA | Dimensionality Reduction |
| 🎯 K-Means | Customer Clustering |
| 🌳 Agglomerative Clustering | Hierarchical Segmentation |
| 📓 Jupyter Notebook | Interactive Development |

</div>

---

# 📁 Project Structure

```text
SmartCart-Customer-Segmentation/
│
├── 📓 SmartCart_Customer_Segmentation_Analysis.ipynb
│
├── 📊 smartcart_customers.csv
│
├── ⚙️ .gitignore
│
└── 📖 README.md
```

---

# 🚀 Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/sumitjhadev/SmartCart-Customer-Segmentation.git
```

## 2️⃣ Navigate to the Project

```bash
cd SmartCart-Customer-Segmentation
```

## 3️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed
```

## 4️⃣ Launch Jupyter

```bash
jupyter lab
```

## 5️⃣ Run the Notebook

Open:

```text
SmartCart_Customer_Segmentation_Analysis.ipynb
```

Run the cells sequentially to reproduce the complete analysis.

---

# 🧠 Skills Demonstrated

This project demonstrates practical experience with:

<div align="center">

### 📊 Data Analysis

Data Exploration • Statistical Analysis • Visualization

### 🧹 Data Preprocessing

Missing Values • Cleaning • Encoding • Scaling

### 🧪 Feature Engineering

Age • Tenure • Spending • Customer-Level Features

### 🤖 Machine Learning

Unsupervised Learning • K-Means • Agglomerative Clustering

### 🧬 Dimensionality Reduction

Principal Component Analysis

### 📈 Model Evaluation

Elbow Method • Knee Detection • Silhouette Analysis

</div>

---

# 🌟 Key Takeaway

> **Customer segmentation is not about assigning arbitrary labels to customers. It is about allowing data to reveal meaningful behavioral patterns.**

SmartCart demonstrates a complete workflow for transforming customer data into structured insights:

```text
RAW DATA
   │
   ▼
PREPROCESSING
   │
   ▼
FEATURE ENGINEERING
   │
   ▼
EXPLORATORY ANALYSIS
   │
   ▼
SCALING + PCA
   │
   ▼
CLUSTERING
   │
   ▼
CUSTOMER SEGMENTS
   │
   ▼
BUSINESS INSIGHTS
```

---

<div align="center">

# 🛒 SmartCart Customer Segmentation

### From Customer Data → Hidden Patterns → Meaningful Insights

<br>

**Built with 🐍 Python • 📊 Data Science • 🤖 Machine Learning**

<br>

⭐ **If you found this project interesting, consider starring the repository!**

<br>

### 👨‍💻 Created by **Sumit Jha**

*Aspiring Data Scientist • Machine Learning Enthusiast*

</div>
