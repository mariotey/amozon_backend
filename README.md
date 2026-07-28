# 📚 Recommendation System

This project implements two widely used types of recommendation algorithms:

### 1. 🤝 Collaborative Filtering

**Collaborative Filtering** recommends items to a user based on the preferences and behaviors of *similar users*. It assumes that if two users liked similar items in the past, they will likely enjoy similar items in the future. Unlike other methods, it does **not** require item metadata and relies entirely on the **user-item interaction matrix** (e.g., ratings, likes, clicks).

### 2. 🧾 Content-Based Filtering

**Content-Based Filtering** recommends items based on the features of the items themselves and a user’s past preferences. It builds a personalized profile by analyzing what the user liked before (e.g., genres, tags, actors in movies) and recommends items with similar characteristics.

### 🔍 Summary

| Feature                     | Collaborative Filtering             | Content-Based Filtering               |
|----------------------------|--------------------------------------|---------------------------------------|
| Based on                   | Similar users' behavior              | Item features + user preferences      |
| Needs item metadata        | ❌ No                                 | ✅ Yes                                 |
| Handles cold start well    | ❌ Poor for new users/items           | ✅ Better for new items                |
| Personalization            | ✅ High                               | ✅ Medium to High                      |
| Novelty in recommendations | ✅ Higher (via peer behavior)         | ❌ Lower (limited to known preferences)|

## 🚀 Setup
This project is designed to run using a standard **Python** installation and **is not intended to be used with Anaconda/Miniconda**.
Please ensure Python is installed and available on your system `PATH` before proceeding.

### 1. Clone the repository

```bash
git clone https://github.com/mariotey/amozon_backend.git
cd amozon_backend
```

### 2. Configure environment variables

Create a `.env` file in the project root directory and populate it with the required Supabase credentials.

For example:

```env
SUPABASE_URL=<your-supabase-url>
SUPABASE_PUBLIC_KEY=<your-supabase-key>
SUPBASE_SECRET_KEY=<your-service-role-key>
```
**NOTE**: Replace the placeholder values with the corresponding credentials from your Supabase project.

### 3. Install dependencies

From the project root directory, run:

```bash
pip install uv

# Install project dependencies
uv sync

# Performs the required project setup
python -m setup
```

### 4. Verify the installation

Build either recommender model to verify that the project has been configured successfully.

Collaborative Filtering:

```bash
python -m collaborative_filtering.main --mode build
```

Content-Based Filtering:

```bash
python -m content_based_filtering.main --mode build
```
