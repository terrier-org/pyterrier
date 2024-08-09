import unittest
import pyterrier as pt, pandas as pd
from .base import BaseTestCase

class TestInit(BaseTestCase):

    def test_set_property(self):
        pt.terrier.set_property("arbitrary.property", 40)
        self.assertEqual("40", pt.terrier.J.ApplicationSetup.appProperties.getProperty("arbitrary.property", "none"))
