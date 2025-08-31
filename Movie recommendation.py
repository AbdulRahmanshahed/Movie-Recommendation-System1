import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Data Preprocessing ----------------
def load_data(path):
    df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\AI&Ml internship\Week 1 Tasks\Task 3\movies.csv")

    # Fill missing values
    for col in ["MOVIES", "GENRE", "ONE-LINE", "STARS"]:
        df[col] = df[col].fillna("")

    # Normalize case
    df["MOVIES"] = df["MOVIES"].str.strip()

    return df

# ---------------- Content-Based Filtering ----------------
def build_content_model(df):
    # Combine descriptive columns into one
    df["content"] = (
        df["GENRE"].astype(str) + " " +
        df["ONE-LINE"].astype(str) + " " +
        df["STARS"].astype(str)
    )

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["content"])

    # Cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend_content(movie_name, df, cosine_sim, top_n=5):
    df = df.reset_index(drop=True)

    matches = df[df["MOVIES"].str.lower() == movie_name.lower()]
    if matches.empty:
        return ["❌ Movie not found in dataset!"]

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    return df[["MOVIES", "GENRE", "RATING", "YEAR"]].iloc[movie_indices]

# ---------------- Popularity-Based (Simulated Collaborative) ----------------
def recommend_popular(df, top_n=5):
    # Weighted score = Rating × log(Votes)
    df["score"] = df["RATING"] * (df["VOTES"] + 1).apply(lambda x: x**0.5)
    top_movies = df.sort_values("score", ascending=False).head(top_n)
    return top_movies[["MOVIES", "GENRE", "RATING", "YEAR"]]

# ---------------- Hybrid Recommendation ----------------
def recommend_hybrid(movie_name, df, cosine_sim, top_n=5):
    content_recs = recommend_content(movie_name, df, cosine_sim, top_n=10)

    if isinstance(content_recs, list):  # if movie not found
        return content_recs

    # Merge with popularity
    content_recs = content_recs.copy()
    merged = pd.merge(content_recs, df[["MOVIES", "VOTES"]], on="MOVIES", how="left")
    merged["score"] = merged["RATING"] * (merged["VOTES"] + 1).apply(lambda x: x**0.5)
    merged = merged.sort_values("score", ascending=False).head(top_n)
    return merged

# ---------------- Main ----------------
def main():
    path = r"C:\Users\DELL\OneDrive\Desktop\AI&Ml internship\Week 1 Tasks\Task 3\movies.csv"
    df = load_data(path)
    cosine_sim = build_content_model(df)

    print("\n🎬 Movie Recommendation System")
    print("1. Content-Based Filtering")
    print("2. Popularity-Based Filtering (simulated collaborative)")
    print("3. Hybrid (Content + Popularity)\n")

    choice = input("Choose option (1/2/3): ").strip()

    if choice == "1":
        movie_name = input("Enter a movie you like: ")
        recs = recommend_content(movie_name, df, cosine_sim, top_n=5)
    elif choice == "2":
        recs = recommend_popular(df, top_n=5)
    elif choice == "3":
        movie_name = input("Enter a movie you like: ")
        recs = recommend_hybrid(movie_name, df, cosine_sim, top_n=5)
    else:
        print("❌ Invalid choice")
        return

    print("\n👉 Recommended Movies:\n")
    print(recs.to_string(index=False))

if __name__ == "__main__":
    main()
