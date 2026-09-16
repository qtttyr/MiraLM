"""Curated demo prompts per domain (for the heatmap + interactive demo)."""

DEMO_PROMPTS: dict[str, list[str]] = {
    "json": [
        "Return a JSON object with a user's id, name, age, city and role: "
        "id 42, name Alice, age 29, city Kyiv, role admin.",
        "Return a JSON object of the event with fields event, day, starts_at "
        "and capacity: event demo_2026, day Tuesday, 10:00, capacity 150.",
        "Return a JSON object with product, price, stock and tags: product "
        "mug, price 500, stock 23, tags [sale, new].",
    ],
    "sql": [
        "Given the schema users(id INTEGER, name TEXT, age INTEGER, city TEXT), "
        "write an SQL query for rows where city = 'Kyiv'.",
        "Given the schema orders(id INTEGER, user_id INTEGER, amount REAL, "
        "status TEXT), write an SQL query to sum amount.",
        "Given the schema products(id INTEGER, name TEXT, price REAL, "
        "stock INTEGER), write an SQL query to count rows where stock < 5.",
    ],
    "cot": [
        "Word problem 1: a shop starts with 12 items and gets 7 more per day "
        "for one day. Total?",
        "Word problem 2: a rectangle is 9 by 8 units. What is its area in "
        "square units?",
        "A farm has 15 chickens and 6 cows. How many legs in total? Solve step by step.",
    ],
}

DEMO_INTRO = (
    "\nMiraLM-47M — structured-output demo (JSON / SQL / CoT)\n"
    + "=" * 52
    + "\nspecial tokens: <|json|> <|sql|> <|cot|> <|think|> <|answer|>\n"
)