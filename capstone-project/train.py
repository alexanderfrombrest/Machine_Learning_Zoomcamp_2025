import pandas as pd
import re
from transform import transform_raw_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

sanctions_data = transform_raw_datasets('datasets/EU-financial-sanctions.csv')

# 1. Build the Vectorizer.
# analyzer='char_wb': creates n-grams only within word boundaries.
# ngram_range=(3, 3): use trigrams as they are industry standard
vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3,3),
    min_df=1, 
    strip_accents='unicode'
)

# 2. Build the sparse matrix
tfidf_matrix = vectorizer.fit_transform(sanctions_data['name_clean'])

# 3. Setup Nearest Neighbors Object
nbrs = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')

# 4. Fit TfIDF matrix into KNN-object
nbrs.fit(tfidf_matrix)


def ngrams(string, n=3):
    string = re.sub(r'[,-./]|',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def fuzzy_search(query):
    # Vectorize the input query
    query_vec = vectorizer.transform([query])
    
    # Find closest neighbors
    distances, indices = nbrs.kneighbors(query_vec)
    
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        dist = distances[0][i]
        # Convert distance (0..1) to Similarity Score (0..100)
        similarity = (1 - dist) * 100
        
        # Filter: Only show relevant matches (e.g. > 50%)
        if similarity > 50:
            match_data = sanctions_data.iloc[idx]
            results.append({
                "Query": query,
                "Match": match_data['name'], # Show original name
                "Score": round(similarity, 1),
                "Source": match_data.get('source', 'Unknown')
            })
            
    return pd.DataFrame(results)

print(fuzzy_search('Sberbank'))