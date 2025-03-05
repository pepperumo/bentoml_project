#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BentoML service for the admissions prediction model.

This service provides:
1. A /login endpoint that returns a JWT for valid credentials.
2. A /predict endpoint that requires a valid JWT token and returns an admission chance prediction.
3. JWT authentication for secure API access.
"""

import bentoml
import numpy as np
import pandas as pd
from bentoml.io import JSON  # Note: bentoml.io is deprecated in v1.4 and will be removed in a future version.
from pydantic import BaseModel, Field
import jwt
import datetime
import os
from typing import Dict

# Load the saved model from BentoML store
admissions_runner = bentoml.sklearn.get("admissions_model:latest").to_runner()

# Create a BentoML service
svc = bentoml.Service("admissions_prediction", runners=[admissions_runner])

# JWT configuration
# In a production environment, these should be stored securely (e.g., environment variables)
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Sample user database (for demo purposes)
users_db = {
    "admin": {
        "username": "admin",
        "password": "admin123",
    },
    "user": {
        "username": "user",
        "password": "pass123",
    }
}

# Models for request/response data validation
class UserCredentials(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class PredictionFeatures(BaseModel):
    GRE_Score: int = Field(..., ge=0, le=340, description="GRE Score (0-340)")
    TOEFL_Score: int = Field(..., ge=0, le=120, description="TOEFL Score (0-120)")
    University_Rating: int = Field(..., ge=1, le=5, description="University Rating (1-5)")
    SOP: float = Field(..., ge=1, le=5, description="Statement of Purpose Strength (1-5)")
    LOR: float = Field(..., ge=1, le=5, description="Letter of Recommendation Strength (1-5)")
    CGPA: float = Field(..., ge=0, le=10, description="Undergraduate CGPA (0-10)")
    Research: int = Field(..., ge=0, le=1, description="Research Experience (0=no, 1=yes)")

    class Config:
        schema_extra = {
            "example": {
                "GRE_Score": 337,
                "TOEFL_Score": 118,
                "University_Rating": 4,
                "SOP": 4.5,
                "LOR": 4.5,
                "CGPA": 9.65,
                "Research": 1
            }
        }

class PredictionResponse(BaseModel):
    chance_of_admit: float

# Helper functions for JWT authentication
def create_access_token(data: dict, expires_delta: datetime.timedelta = None):
    """
    Create a JWT access token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str):
    """
    Decode and verify JWT token.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return None
        return username
    except jwt.PyJWTError:
        return None

def verify_token(auth_header: str):
    """
    Verify JWT token from the Authorization header.
    """
    if not auth_header:
        return None
    
    if not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.split("Bearer ")[1]
    username = decode_token(token)
    
    if username is None:
        return None
        
    if username not in users_db:
        return None
        
    return username

# Service endpoints
@svc.api(input=JSON(pydantic_model=UserCredentials), output=JSON(pydantic_model=Token))
def login(user_credentials: UserCredentials) -> Dict:
    """
    Login endpoint that provides a JWT token for valid credentials.
    """
    # Check if the user exists and the password is correct
    user = users_db.get(user_credentials.username)
    if not user or user_credentials.password != user["password"]:
        raise bentoml.exceptions.BentoMLException(
            status_code=401,
            message="Incorrect username or password"
        )
    
    # Create access token
    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_credentials.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@svc.api(input=JSON(pydantic_model=PredictionFeatures), output=JSON(pydantic_model=PredictionResponse))
async def predict(features: PredictionFeatures, context: bentoml.Context) -> Dict:
    """
    Prediction endpoint that requires JWT authentication.
    """
    # Check for JWT token in request headers
    auth_header = context.http_headers.get("Authorization")
    username = verify_token(auth_header)
    if username is None:
        raise bentoml.exceptions.BentoMLException(
            status_code=401,
            message="Authentication failed. Please provide a valid JWT token."
        )
    
    # Convert features to a DataFrame for prediction
    input_data = pd.DataFrame([{
        "GRE Score": features.GRE_Score,
        "TOEFL Score": features.TOEFL_Score,
        "University Rating": features.University_Rating,
        "SOP": features.SOP,
        "LOR": features.LOR,  # Corrected column name (removed extra space)
        "CGPA": features.CGPA,
        "Research": features.Research
    }])
    
    # Make prediction asynchronously
    result = await admissions_runner.predict.async_run(input_data)
    prediction = float(result[0])
    # Ensure prediction is between 0 and 1
    prediction = max(0, min(1, prediction))
    
    return {"chance_of_admit": prediction}

# Health check endpoint
@svc.api(input=JSON(), output=JSON())
def health(_):
    """
    Health check endpoint.
    
    Args:
        _: Dummy parameter to satisfy BentoML API requirements.
        
    Returns:
        A JSON object indicating the service status.
    """
    return {"status": "healthy"}
