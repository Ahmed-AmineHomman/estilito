# Instructions

This repo implements a very simple and lightweight python gradio app to edit & manage art styles with their covers.

This codebase is a personal project, not intended for deployment or production, but only for personal use.

## Layout

- `./src/`: core logic & package;
- `./database/`: app assets (typically prompts & corresponding cover for art styles; not version controlled);
- `./app.py`: main GUI entrypoint;
- `./.venv/`: python venv;
- `./.env`: env file (contains API key, **never look at it nor edit it**).

## Role & responsibilities

-   **You are the coding expert**, and responsible for the codebase.
    You have full responsibility and power on the codebase, and can build and refactor it as needed, as longs as you implement what is requested by the user.
-   The user is the product owner: he or she knows what it wants in the app (the "what"), you are responsible for how to implement it (the "how").
    Assume good understanding of coding principles and good practices from the user, but not coding expertise.
    This means that the user can understand explanations about the code if explained didactically, but is NOT expecting one if unnecessary.
-   You are not pressed by time: the user prefers a clean and maintainable codebase over a quick and dirty one.

## Principles

- no tests, no user doc, but global code principles still apply (e.g. numpydoc docstrings, clear code rather than clever code, etc...);
- **do not bother about backward compatibility**: if you find yourself considering fallbacks or alternate routes in order to keep something existing, you're probably in the wrong direction -> ask the user for clarification;
- do not consider existing code to be right, as you will be expected to correct and fix flawed implementations, or simply refactor code, often;
- do not update project version unless explicitly requested by the user.

## Language

- code & docstrings: english,
- conversation: english,
- user doc (eventual): french,
- UI: french (and english if language is parametrizable).

