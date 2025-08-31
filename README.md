This project is a **Content-Based Movie Recommendation System** that suggests movies similar to a given title.  
It uses **TF-IDF Vectorization + Cosine Similarity** to recommend movies based on `GENRE`, `STARS`, and `ONE-LINE`.

---

## 📂 Dataset
- Single CSV file: `movies.csv`  
- Columns:
  - `MOVIES` → Movie title  
  - `YEAR` → Release year  
  - `GENRE` → Movie genres  
  - `RATING` → IMDb rating or given rating  
  - `ONE-LINE` → Short description / tagline  
  - `STARS` → Main actors  
  - `VOTES` → Number of votes  
  - `RunTime` → Duration of the movie  
  - `Gross` → Revenue  

Example structure:
movies.csv
│── MOVIES, YEAR, GENRE, RATING, ONE-LINE, STARS, VOTES, RunTime, Gross
│── Inception, 2010, Action|Sci-Fi, 8.8, "Your mind is the scene of the crime", Leonardo DiCaprio, 2000000, 148, 829895144
│── Interstellar, 2014, Adventure|Drama|Sci-Fi, 8.6, "Mankind was born on Earth. It was never meant to die here.", Matthew McConaughey, 1800000, 169, 677471339
│── The Dark Knight, 2008, Action|Crime|Drama, 9.0, "Why so serious?", Christian Bale, 2500000, 152, 1004558444

yaml
Copy code

---

## ⚙️ Features
- Cleans and preprocesses dataset (removes nulls, standardizes text).  
- Combines `GENRE + STARS + ONE-LINE` into a single feature.  
- Uses **TF-IDF Vectorization** and **Cosine Similarity**.  
- Input: Movie name → Output: Top 5 recommended movies.  

---

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
Install dependencies:

bash
Copy code
pip install pandas scikit-learn
Place your dataset:

Copy code
movies.csv
Run the script:

bash
Copy code
python "Movie recommendation.py"
🎥 Example
After running the script:
