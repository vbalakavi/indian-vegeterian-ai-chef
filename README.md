# Indian Vegeterian AI Chef

This is the clean publish-ready Streamlit folder for the app.

## Included Files

- `app.py`
- `agent.py`
- `config.py`
- `requirements.txt`
- `db/`
- `.env.example`
- `.streamlit/secrets.toml.example`

## Deploy To Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. In Streamlit, create a new app and point it to `app.py`.
3. In the Streamlit app settings, add these secrets:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
APP_USERNAME = "your_username"
APP_PASSWORD = "your_password"
```

## Local Run

1. Copy `.env.example` to `.env`
2. Fill in the values
3. Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Do not commit your real `.env`
- The `db/` folder must stay with the app because the agent loads the FAISS index from it
