# ==========================================================
# Flask Web Application Integration Tests Suite
# ==========================================================

import os
import sys
import unittest

# Ensure the root directory is available in Python PATH for module lookup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Clean, native package import matching your 'flask_app/' folder name
from flask_app.app import app


class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Configure Flask test client with test configuration flags.
        """
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

    # ==========================================================
    # 1. UI Endpoint Tests
    # ==========================================================

    def test_home_page_loads_successfully(self):
        """
        Verify that the root endpoint '/' returns a 200 HTTP status code 
        and renders the expected HTML title element.
        """
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200, "Home page failed to return HTTP 200.")
        self.assertIn(
            b'<title>Sentiment Analysis</title>', 
            response.data, 
            "Title element missing from rendered HTML."
        )

    # ==========================================================
    # 2. Prediction Inference Endpoint Tests
    # ==========================================================

    def test_predict_positive_sentiment(self):
        """
        Test end-to-end inference POST request for positive sentiment input.
        """
        response = self.client.post(
            '/predict', 
            data={'text': 'I absolutely love this product! It works so well and brings me huge joy.'},
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200, "Prediction endpoint failed with non-200 status.")
        self.assertIn(
            b'Happy', 
            response.data, 
            "Expected 'Happy' prediction result in response HTML."
        )

    def test_predict_negative_sentiment(self):
        """
        Test end-to-end inference POST request for negative sentiment input.
        """
        response = self.client.post(
            '/predict', 
            data={'text': 'This is terrible and worst experience ever, completely broken and bad.'},
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200, "Prediction endpoint failed with non-200 status.")
        self.assertIn(
            b'Sad', 
            response.data, 
            "Expected 'Sad' prediction result in response HTML."
        )

    def test_predict_empty_input_handling(self):
        """
        Verify that submitting blank input text displays user warning message gracefully.
        """
        response = self.client.post(
            '/predict', 
            data={'text': '   '},
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b'Please enter text.', 
            response.data, 
            "Expected validation warning message on empty input string."
        )


if __name__ == '__main__':
    unittest.main()