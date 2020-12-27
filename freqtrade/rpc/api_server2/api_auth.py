import secrets
from datetime import datetime, timedelta

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.http import HTTPBasic, HTTPBasicCredentials

from freqtrade.rpc.api_server2.api_models import AccessAndRefreshToken, AccessToken

from .deps import get_config


SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

router_login = APIRouter()


def verify_auth(config, username: str, password: str):
    """Verify username/password"""
    return (secrets.compare_digest(username, config['api_server'].get('username')) and
            secrets.compare_digest(password, config['api_server'].get('password')))


httpbasic = HTTPBasic(auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def get_user_from_token(token, token_type: str = "access"):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        if payload.get("type") != token_type:
            raise credentials_exception

    except jwt.PyJWTError:
        raise credentials_exception
    return username


def create_token(data: dict, token_type: str = "access") -> str:
    to_encode = data.copy()
    if token_type == "access":
        expire = datetime.utcnow() + timedelta(minutes=15)
    elif token_type == "refresh":
        expire = datetime.utcnow() + timedelta(days=30)
    else:
        raise ValueError()
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": token_type,
    })
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def http_basic_or_jwt_token(form_data: HTTPBasicCredentials = Depends(httpbasic),
                            token: str = Depends(oauth2_scheme), config=Depends(get_config)):
    if token:
        return get_user_from_token(token)
    elif form_data and verify_auth(config, form_data.username, form_data.password):
        return form_data.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
    )


@router_login.post('/token/login', response_model=AccessAndRefreshToken)
def token_login(form_data: HTTPBasicCredentials = Depends(HTTPBasic()), config=Depends(get_config)):

    if verify_auth(config, form_data.username, form_data.password):
        token_data = {'sub': form_data.username}
        access_token = create_token(token_data)
        refresh_token = create_token(token_data, token_type="refresh")
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
def token_refresh(token: str = Depends(oauth2_scheme)):
    # Refresh token
    u = get_user_from_token(token, 'refresh')
    token_data = {'sub': u}
    access_token = create_token(token_data, token_type="access")
    return {'access_token': access_token}
