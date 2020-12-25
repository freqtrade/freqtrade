from freqtrade.rpc.api_server2.models import AccessAndRefreshToken, AccessToken
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security.http import HTTPBasic, HTTPBasicCredentials
from fastapi.security.utils import get_authorization_scheme_param
from fastapi_jwt_auth import AuthJWT
from pydantic import BaseModel

from .deps import get_config


SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

router_login = APIRouter()


class Settings(BaseModel):
    # TODO: should be set as config['api_server'].get('jwt_secret_key', 'super-secret')
    authjwt_secret_key: str = "secret"


@AuthJWT.load_config
def get_jwt_config():
    return Settings()


def verify_auth(config, username: str, password: str):
    return (secrets.compare_digest(username, config['api_server'].get('username')) and
            secrets.compare_digest(password, config['api_server'].get('password')))


class HTTPBasicOrJWTToken(HTTPBasic):
    description = "Token Or Pass auth"

    async def __call__(self, request: Request, config=Depends(get_config)  # type: ignore
                       ) -> Optional[str]:
        header_authorization: str = request.headers.get("Authorization")
        header_scheme, header_param = get_authorization_scheme_param(header_authorization)
        if header_scheme.lower() == 'bearer':
            AuthJWT.jwt_required()
        elif header_scheme.lower() == 'basic':
            credentials: Optional[HTTPBasicCredentials] = await HTTPBasic()(request)
            if credentials and verify_auth(config, credentials.username, credentials.password):
                return credentials.username
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )


@router_login.post('/token/login', response_model=AccessAndRefreshToken)
def token_login(form_data: HTTPBasicCredentials = Depends(HTTPBasic()), config=Depends(get_config)):

    print(form_data)
    Authorize = AuthJWT()

    if verify_auth(config, form_data.username, form_data.password):
        token_data = form_data.username
        access_token = Authorize.create_access_token(subject=token_data)
        refresh_token = Authorize.create_refresh_token(subject=token_data)
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )


@router_login.post('/token/refresh', response_model=AccessToken)
def token_refresh(Authorize: AuthJWT = Depends()):
    Authorize.jwt_refresh_token_required()

    access_token = Authorize.create_access_token(subject=Authorize.get_jwt_subject())
    return {'access_token': access_token}
