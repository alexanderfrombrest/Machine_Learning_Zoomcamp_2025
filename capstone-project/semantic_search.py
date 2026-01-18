import pandas as pd
from sentence_transformers import SentenceTransformer, util

hs_data = pd.read_csv('datasets/harmonized-system.csv', encoding='utf-8')

hs_codes = hs_data['hscode'].to_list()
hs_names_list = hs_data['description'].to_list()

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Pre-Calculate embeddings by calling model.encode()
# We store them so we don't have to re-calculate them every time.
hs_embeddings = model.encode(hs_names_list, convert_to_tensor=True)
# print(embeddings.shape)
# (6940, 384)

# 3. Auditor function 
def validate_classification(product_desc, assigned_hs_code):
    """
    Checks if a product description matches the 'official' definition 
    of its assigned HS code.
    """
    assigned_hs_code_long = str(assigned_hs_code).strip()

    # If your input is 8 or 10 digits (National level), we must truncate to 6 to find a match.
    assigned_hs_code = assigned_hs_code_long[:6]

    if assigned_hs_code not in hs_codes:
        return {
            "Product": product_desc,
            "Assigned_HS": assigned_hs_code, 
            "Risk_Level": "UNKNOWN_CODE", 
            "Reason": f"Base code {assigned_hs_code} not found in master data"
        }
    
    # 1. Vectorize the user's product description
    product_vec = model.encode(product_desc, convert_to_tensor=True)

    # 2. Get the vector for official description of the HS code
    idx = hs_codes.index(assigned_hs_code)
    official_vec = hs_embeddings[idx]

    # 3. Calculate Cosine Similarity
    score = util.cos_sim(product_vec, official_vec).item()

    # 4. Define Risk Logic
    if score < 0.25:
        risk = "HIGH (MISCLASSIFICATION)"
    elif score < 0.5:
        risk = "MEDIUM (REVIEW NEEDED)"
    else:
        risk = "LOW (MATCH CONFIRMED)"

    official_desc = hs_names_list[idx]

    return {
        "Product": product_desc,
        "Assigned_HS": assigned_hs_code,
        "Official_Desc": official_desc[:50] + "...",
        "Similarity_Score": round(score, 3),
        "Risk_Level": risk
    }

# ----------------------------------------------------
# 4. Test Cases
# ----------------------------------------------------
test_cases = [
    ("RUBBER AND SILICON HOSES 22B 250X4", "39174000"),
    ("PVC PIPE", "39171020"),
    ("POLO SHORTY BODY", "39173990"),
    ("FRP ECCENTRIC REDUCER ITEM CODE 1201001192118", "39172990")
]

print("\n--- AUDIT RESULTS ---")
results = []
for desc, code in test_cases:
    res = validate_classification(desc, code)
    results.append(res)

# Display nicely
df_results = pd.DataFrame(results)
print(df_results[['Product', 'Assigned_HS', 'Official_Desc', 'Similarity_Score', 'Risk_Level']])

