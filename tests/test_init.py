import unittest
import pyterrier as pt, pandas as pd
from .base import BaseTestCase

class TestInit(BaseTestCase):

    def test_set_property(self):
        pt.set_property("arbitrary.property", 40)
        self.assertEqual("40", pt.ApplicationSetup.appProperties.getProperty("arbitrary.property", "none"))
