import unittest
import pandas as pd
from src.analysis.bias_detection import BiasDetector

class TestBiasDetection(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset for testing
        data = {
            'gender': ['male', 'female', 'female', 'male', 'female'],
            'income': [50000, 60000, 70000, 80000, 90000],
            'target': [1, 0, 1, 0, 1]
        }
        self.dataset = pd.DataFrame(data)
        self.detector = BiasDetector(self.dataset)
    
    def test_statistical_parity(self):
        result = self.detector.calculate_statistical_parity()
        self.assertIn('gender', result)
    
    def test_disparate_impact(self):
        result = self.detector.calculate_disparate_impact()
        self.assertIn('gender', result)
    
    def test_equal_opportunity(self):
        result = self.detector.calculate_equal_opportunity()
        self.assertIn('gender', result)

if __name__ == '__main__':
    unittest.main()