import os
import threading

import uvicorn
from pyngrok import ngrok

PORT = int(os.getenv("PORT", "8000"))
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")

if NGROK_AUTHTOKEN:
    ngrok.set_auth_token(NGROK_AUTHTOKEN)

public_url = ngrok.connect(PORT, bind_tls=True).public_url

print()
print(f"Public ngrok URL: {public_url}")
print(f"Swagger docs: {public_url}/docs")
print()
print("Backend routes:")
print(f"Price prediction: POST {public_url}/price/predict")
print(f"Court demand prediction: POST {public_url}/court-demand/predict")
print(f"Matchmaking pair prediction: POST {public_url}/matchmaking/predict-pair")
print(f"Open match recommendation: POST {public_url}/matchmaking/recommend-open-match")
print()

threading.Thread(
    target=lambda: uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False),
    daemon=False,
).start()
