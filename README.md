# 🧾 GOATED: Best Content Aggregator Web App

> *“Only the Best. Always.”*

---

## 🧠 Overview

GOATED is a content discovery web app that allows users to search for the best items across various domains (TV shows, books, courses, products, etc.) by aggregating and averaging ratings from multiple trusted sources (e.g., IMDb, Rotten Tomatoes, Goodreads, Amazon). The result is a curated list of only the top-rated ("GOATED") items.

---

## 🎯 Core Goals

* 🧠 Search the internet for reputable ratings and reviews
* 📊 Aggregate and normalize scores from multiple sources
* 🏆 Present a clean, ranked list of top items
* 💾 save/bookmark GOATED items and lists
* 🔎 Allow advanced search and filtering by category, year, genre, etc.

---

## 🧩 Features

### 1. **Search**

* Input: free-form text (e.g., “best thriller TV shows 2023”)
* Backend detects **category**, **keywords**, and **filters**
* Returns a list of results sorted by GOATED score

### 2. **GOATED Score Aggregation**

* Fetch data from multiple trusted sources
* Normalize all scores (e.g., 8.5/10, 90%, 4.5/5 → 85%)
* Compute average GOATED score

### 3. **Data Sources**

Per category:

| Category  | Sources                                       |
|-----------|-----------------------------------------------|
| TV/Movies | IMDb, Rotten Tomatoes, Metacritic, Letterboxd |
| Books     | Goodreads, Amazon, NYTimes                    |
| Courses   | Udemy, Coursera, edX, Class Central           |
| Products  | Amazon, Wirecutter, TechRadar, CNET           |

### 4. **Frontend**

* Clean search interface
* Responsive GOATED list UI
* Filter by:

  * Category
  * Rating
  * Year
  * Platform/source
* Each item card shows:

  * Title & image
  * Individual ratings from each source
  * GOATED average
  * Description
  * Source links

### 5. **User Accounts**

* Bookmark/save GOATED items
* Personalized lists
* Search history

---

## ⚙️ Tech Stack

### Frontend

* **Next.js (React)**
* **Tailwind CSS** (for styling)
* Axios (for API calls)

### Backend

* **FastAPI**: core API layer, handles user queries
* **httpx (async)**: async external API requests: IMDb, Udemy, Reddit, etc.
* **pandas (with numpy)**: normalize ratings & compute GOATED score
* **spaCy**, Hugging Face Transformers: detect categories, filters, keywords
* Selectolax, Playwright: scraping static or JS-heavy sites

### Data / Caching

* **Redis** – (cache API responses, avoid rate limits).
* PostgreSQL – only if you need persistence.

### Hosting
* Frontend: Vercel or Netlify
* Backend: Render, Railway, Heroku
* Database: Supabase, ElephantSQL

---

## 🧱 Database Schema (simplified)

### `Item`

| Field         | Type                            |
| ------------- | ------------------------------- |
| id            | UUID                            |
| title         | string                          |
| category      | string (TV, Book, Course, etc.) |
| year          | int                             |
| image\_url    | string                          |
| description   | text                            |
| goated\_score | float (0–100)                   |
| sources       | FK → `Rating` (many-to-one)     |

### `Rating`

| Field             | Type         |
| ----------------- | ------------ |
| id                | UUID         |
| item\_id          | FK to `Item` |
| source\_name      | string       |
| raw\_score        | string       |
| normalized\_score | float        |
| source\_url       | string       |

---

## 🔁 Workflow

1. User searches:“best AI courses 2024”
2. Backend extracts category, keywords, filters
3. Data fetched via APIs/scrapers (Coursera, Udemy, etc.)
4. Ratings normalized & averaged
5. GOATED list ranked & returned as JSON
6. Frontend displays cards sorted by GOATED score

---

## 🧪 MVP Scope

* [x] Search UI
* [x] Django backend + DRF API
* [x] Scrapers for 1–2 categories (e.g., IMDb, Goodreads)
* [x] Score normalization + ranking logic
* [x] Item list UI with GOATED scores

---

## 🌱 Future Features

* Reddit as a data source
* Personalized lists
* Real-time updates
* AI-powered query understanding
* Chrome Extension, Desktop App and/or Mobile App
* Trending GOATED content
* “Battle of the GOATs” (compare any two items)

---
