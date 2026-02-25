import os
from pathlib import Path
import httpx
from fastapi import HTTPException
from chainlit.user import User
from chainlit.oauth_providers import OAuthProvider
from .validate_jwt import validate_jwt, decode_jwt


class AuthentikOAuthProvider(OAuthProvider):
    id = "authentik"
    env = [
        "OAUTH_AUTHENTIK_URL_BASE",
        "OAUTH_AUTHENTIK_CLIENT_ID",
        "OAUTH_AUTHENTIK_CLIENT_SECRET",
        "OAUTH_APPLICATION_NAME",
        "OAUTH_AUTHENTIK_PRIVATE_KEY_PATH",
    ]
    authorize_params = {
        "response_type": "code",
        "scope": "openid profile email",
        "response_mode": "query",
    }

    url_base = ""
    client_id = ""
    client_secret = ""
    application_name = ""
    authorize_url = ""
    token_url = ""
    iss_url = ""
    jwks_url = ""

    def __init__(self):
        self.url_base = self._require_env_var("OAUTH_AUTHENTIK_URL_BASE").strip("/")
        self.client_id = self._require_env_var("OAUTH_AUTHENTIK_CLIENT_ID")
        self.client_secret = self._require_env_var("OAUTH_AUTHENTIK_CLIENT_SECRET")
        self.application_name = self._require_env_var("OAUTH_APPLICATION_NAME")

        self.authorize_url = f"{self.url_base}/authorize/"
        self.token_url = f"{self.url_base}/token/"
        self.iss_url = f"{self.url_base}/{self.application_name}/"
        self.jwks_url = f"{self.url_base}/{self.application_name}/jwks/"

        private_key_path = Path(self._require_env_var("OAUTH_AUTHENTIK_PRIVATE_KEY_PATH"))

        if not private_key_path.exists():
            raise FileNotFoundError(f"Private key file not found at {private_key_path}")

        self.private_key = private_key_path.read_text().strip()

    @staticmethod
    def _require_env_var(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise ValueError(f"Missing required environment variable: {name}")
        return value

    async def get_token(self, code: str, url: str) -> str:
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": url,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.token_url, data=payload)
            response.raise_for_status()
            response_json = response.json()

            token = response_json.get("id_token")
            if not token:
                raise HTTPException(status_code=400, detail=f"Token response missing 'id_token': {response_json}")

            return token

    async def get_user_info(self, token: str):
        async with httpx.AsyncClient() as client:
            key, token = validate_jwt(token, jwks_uri=self.jwks_url, private_key=self.private_key)
            authentik_user = decode_jwt(
                token,
                key,
                audience=self.client_id,
                issuer=self.iss_url,
            )

            try:
                user = User(
                    identifier=authentik_user["email"][0] if "emails" in authentik_user else authentik_user["email"],
                    display_name=authentik_user.get("name", authentik_user.get("preferred_username", "User")),
                )
                return authentik_user, user
            except Exception as e:
                raise HTTPException(status_code=400, detail="Failed to get the user info")
