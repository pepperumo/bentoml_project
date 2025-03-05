#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the BentoML service.

Tests include:
1. Authentication flow (login endpoint)
2. JWT token validation
3. Prediction endpoint with various scenarios
"""

import os
import sys
import unittest
import json
import time
import jwt
import requests
from unittest.mock import patch, MagicMock

# Add project root to python path to allow importing the service module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import service module (will be imported but not run until tests)
from src import service

class TestAdmissionsService(unittest.TestCase):
    """Test cases for the Admissions Prediction Service."""

    # Service URL - can be overridden with environment variable
    service_url = os.environ.get("SERVICE_URL", "http://localhost:3000")
    
    # Test user credentials
    valid_user = {
        "username": "admin",
        "password": "admin123"
    }
    
    invalid_user = {
        "username": "admin",
        "password": "wrongpassword"
    }
    
    # Sample prediction data
    valid_prediction_data = {
        "GRE_Score": 337,
        "TOEFL_Score": 118,
        "University_Rating": 4,
        "SOP": 4.5,
        "LOR": 4.5,
        "CGPA": 9.65,
        "Research": 1
    }
    
    invalid_prediction_data = {
        "GRE_Score": 400,  # Invalid: higher than maximum allowed
        "TOEFL_Score": 118,
        "University_Rating": 4,
        "SOP": 4.5,
        "LOR": 4.5,
        "CGPA": 9.65,
        "Research": 1
    }
    
    def setUp(self):
        """Setup before each test."""
        # Create a valid token for testing
        self.valid_token = service.create_access_token({"sub": "admin"})
        
        # Create an expired token for testing
        exp_time = time.time() - 3600  # 1 hour in the past
        payload = {"sub": "admin", "exp": exp_time}
        self.expired_token = jwt.encode(
            payload, 
            service.SECRET_KEY, 
            algorithm=service.ALGORITHM
        )

    def get_headers(self, with_token=True, expired=False):
        """Helper to get request headers with/without token."""
        headers = {"Content-Type": "application/json"}
        if with_token:
            token = self.expired_token if expired else self.valid_token
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @patch('requests.post')
    def test_login_success(self, mock_post):
        """Test successful login."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "mocked_token", "token_type": "bearer"}
        mock_post.return_value = mock_response
        
        # Make request
        response = requests.post(
            f"{self.service_url}/login",
            json=self.valid_user,
            headers={"Content-Type": "application/json"}
        )
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["token_type"], "bearer")

    @patch('requests.post')
    def test_login_failure(self, mock_post):
        """Test failed login with invalid credentials."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"detail": "Incorrect username or password"}
        mock_post.return_value = mock_response
        
        # Make request
        response = requests.post(
            f"{self.service_url}/login",
            json=self.invalid_user,
            headers={"Content-Type": "application/json"}
        )
        
        # Assertions
        self.assertEqual(response.status_code, 401)

    @patch('requests.post')
    def test_predict_success(self, mock_post):
        """Test successful prediction with valid token."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"chance_of_admit": 0.92}
        mock_post.return_value = mock_response
        
        # Make request
        response = requests.post(
            f"{self.service_url}/predict",
            json=self.valid_prediction_data,
            headers=self.get_headers(with_token=True)
        )
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("chance_of_admit", data)
        self.assertIsInstance(data["chance_of_admit"], float)
        self.assertTrue(0 <= data["chance_of_admit"] <= 1)

    @patch('requests.post')
    def test_predict_no_token(self, mock_post):
        """Test prediction without token."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"detail": "Authentication failed"}
        mock_post.return_value = mock_response
        
        # Make request
        response = requests.post(
            f"{self.service_url}/predict",
            json=self.valid_prediction_data,
            headers=self.get_headers(with_token=False)
        )
        
        # Assertions
        self.assertEqual(response.status_code, 401)

    @patch('requests.post')
    def test_predict_expired_token(self, mock_post):
        """Test prediction with expired token."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"detail": "Authentication failed"}
        mock_post.return_value = mock_response
        
        # Make request
        response = requests.post(
            f"{self.service_url}/predict",
            json=self.valid_prediction_data,
            headers=self.get_headers(expired=True)
        )
        
        # Assertions
        self.assertEqual(response.status_code, 401)

    @patch('requests.post')
    def test_predict_invalid_data(self, mock_post):
        """Test prediction with invalid data."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.json.return_value = {"detail": "Validation error"}
        mock_post.return_value = mock_response
        
        # Make request
        response = requests.post(
            f"{self.service_url}/predict",
            json=self.invalid_prediction_data,
            headers=self.get_headers()
        )
        
        # Assertions
        self.assertEqual(response.status_code, 422)

    @patch('requests.get')
    def test_health(self, mock_get):
        """Test health endpoint."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response
        
        # Make request
        response = requests.get(f"{self.service_url}/health")
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

if __name__ == "__main__":
    unittest.main()