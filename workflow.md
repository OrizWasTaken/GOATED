# 🧑‍💻 GOATED Project Workflow

This doc explains **how we organize work and version control** for the GOATED app. Since this is a passion project, the workflow is lightweight but still structured enough to keep code clean.

---

## 🔹 Branching Strategy

* **`main`** → always stable, working version (demo-ready).
* **Feature branches** → one per feature/section.

  * Example names:

    * `backend-search`
    * `backend-scrapers`
    * `backend-aggregation`
    * `frontend-ui`
    * `frontend-item-card`
    * `database`
    * `cache`
    * `tests`

Workflow:

1. Create a branch for a feature:

   ```bash
   git checkout -b backend-search
   ```
2. Commit small, logical steps (see below).
3. Merge into `main` once the feature works:

   ```bash
   git checkout main
   git merge backend-search
   ```

---

## 🔹 Commit Guidelines

Keep commits **small and descriptive**. Format:

```
<type>: <short description>
```

Examples:

* `feat: add FastAPI search endpoint`
* `feat: implement category + keyword detection with spaCy`
* `feat: compute normalized GOATED score`
* `fix: handle missing IMDb data`
* `style: apply Tailwind styling to item cards`
* `test: add unit tests for aggregation logic`

Types we use:

* `feat:` → new feature
* `fix:` → bug fix
* `chore:` → maintenance (configs, deps, etc.)
* `style:` → UI/design changes
* `test:` → adding/updating tests
* `docs:` → documentation updates

---

## 🔹 Milestones / Tags

We use **tags** to mark important stages:

```bash
git tag -a v0.1 -m "MVP: Search + IMDb + Goodreads + GOATED score"
git tag -a v0.2 -m "Frontend UI with item cards"
git tag -a v1.0 -m "Database + caching + tests complete"
git push origin --tags
```

---

## 🔹 Development Flow

1. Pick the next feature (small, focused).
2. Create a branch.
3. Code + commit small changes.
4. Test locally.
5. Merge back to `main`.
6. Tag milestone if needed.

---

## 🔹 Example Roadmap

* `backend-search` → FastAPI + category detection (spaCy)
* `backend-scrapers` → IMDb + Goodreads fetchers (MVP scope)
* `backend-aggregation` → normalize ratings + compute GOATED score
* `frontend-ui` → search bar + results page
* `frontend-item-card` → card UI (title, image, ratings, GOATED avg)
* `database` → Item + Rating schema (Postgres/SQLAlchemy)
* `cache` → Redis caching for API calls
* `tests` → unit + integration tests