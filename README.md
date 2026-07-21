# Digital Wallet Backend

A payment/wallet backend built for learning real fintech backend engineering:
API design, database schema design, authentication, and testing.

## Design notes
This uses a **ledger pattern**: every deposit/withdrawal creates a `Transaction`
row, and an account's balance is calculated as the sum of its transactions —
never stored and overwritten directly. This is how real financial systems
work: it gives you a full audit trail for free, and makes it much harder to
"lose" money to a bug or race condition.

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the API
```powershell
uvicorn app.main:app --reload
```
Then open **http://127.0.0.1:8000/docs** — FastAPI auto-generates interactive
API documentation where you can try every endpoint directly in the browser.

## Run the tests
```powershell
pytest -v
```
Should show 1 passed — a full signup → login → deposit → withdraw flow.

## Try it manually (via /docs or curl)
1. `POST /auth/signup` with `{"email": "you@example.com", "password": "yourpassword123"}`
2. `POST /auth/login` with form fields `username` (your email) and `password`
   → copy the `access_token` from the response
3. Click "Authorize" in `/docs` (or add header `Authorization: Bearer <token>`)
   and paste the token
4. `GET /accounts/me` → note your `id`
5. `POST /accounts/{id}/deposit` with `{"amount": 100}`
6. `GET /accounts/me` again → balance should now show 100

---

## Week 1 checklist (what's already done, and what's next)

- [x] Database schema: users, accounts, transactions (ledger pattern)
- [x] Signup / login with JWT auth
- [x] Deposit endpoint
- [x] Withdraw endpoint (with insufficient-balance check)
- [x] Balance is derived from transaction history, not stored directly
- [x] Transaction history endpoint
- [x] First passing test

### Try these yourself to deepen Week 1 before moving to Week 2:
- [ ] Add a `GET /transactions/{id}` endpoint to fetch a single transaction by ID
- [ ] Add pagination to `GET /accounts/{id}/transactions` (`?limit=10&offset=0`)
- [ ] Write a test for the "withdraw more than balance" failure case explicitly
- [ ] Write a test for signup with a duplicate email (should 400)
- [ ] Add a `.env` file + `python-dotenv` so `JWT_SECRET_KEY` isn't hardcoded

## Week 2 preview: transfers between accounts
The trickiest part of Week 2 is making a transfer **atomic** — both the debit
from account A and the credit to account B must succeed together, or neither
should happen. This is usually done inside a single database transaction
(commit both rows together, or roll back both on any error). We'll build
`POST /accounts/{id}/transfer` together when you're ready.
