#importing libraries

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# reading data

amazon_reviews = pd.read_csv('C:\\Users\\gulshan\\Desktop\\project\\AllProductReviews.csv')
amazon_reviews_lines = amazon_reviews.groupby('Product').count()
amazon_reviews_lines = amazon_reviews_lines.sort_values(by=['ReviewTitle'], ascending=False)[:10]
top_product_names = amazon_reviews_lines.index.values

# filtering out non-top products

df_review_analysis = amazon_reviews[amazon_reviews['Product'].isin(top_product_names)]
df_review_analysis = df_review_analysis[[ 'Product','ReviewTitle']]

# calculating review score

sid = SentimentIntensityAnalyzer()
df_review_analysis.reset_index(inplace=True, drop=True)
df_review_analysis[['neg', 'neu', 'pos', 'compound']] = df_review_analysis['ReviewTitle'].apply(sid.polarity_scores).apply(pd.Series)
df_review_analysis

#taking user input and sorting top 10 positive and negative reviews

product_name=input("enter the product name:")
most_popular=df_review_analysis[df_review_analysis["Product"] == product_name]
popular_products=most_popular.sort_values('compound', ascending=False)
print("\nTop 10 Positve Reviews")
popular_products.head(10)

print("\nTop 10 Negative Reviews")
popular_products.tail(10)
