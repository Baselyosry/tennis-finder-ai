import os

import uvicorn
from pyngrok import ngrok

PORT = int(os.getenv("PORT", "8000"))
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")


def main() -> None:
    if NGROK_AUTHTOKEN:
        ngrok.set_auth_token(NGROK_AUTHTOKEN)

    tunnel = ngrok.connect(PORT, bind_tls=True)
    public_url = tunnel.public_url

    print()
    print(f"Public ngrok URL: {public_url}", flush=True)
    print(f"Swagger docs: {public_url}/docs", flush=True)
    print()
    print("Backend routes:", flush=True)
    print(f"Price prediction: POST {public_url}/price/predict", flush=True)
    print(f"Court demand prediction: POST {public_url}/court-demand/predict", flush=True)
    print(f"Matchmaking pair prediction: POST {public_url}/matchmaking/predict-pair", flush=True)
    print(f"Open match recommendation: POST {public_url}/matchmaking/recommend-open-match", flush=True)
    print()

    try:
        uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
    finally:
        ngrok.disconnect(public_url)
        ngrok.kill()


if __name__ == "__main__":
    main()
