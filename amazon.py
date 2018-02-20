from __future__ import print_function

from amazon_scraper import AmazonScraper
import itertools
import bottlenose
import time
from bs4 import BeautifulSoup
from urllib.error import HTTPError
import random

def error_handler(err):
    ex = err['exception']
    if isinstance(ex, HTTPError) and ex.code == 503:
        time.sleep(random.expovariate(0.1))
        return True

amzn = AmazonScraper(
    'AKIAJW7QUMZFYCKRJU5A', 'LjZSLvVpQ49eL+6At3BhKXvagr+IIdnmA3BINMqu', 'absingh0f-21', Region='IN',MaxQPS=0.9,
    Timeout=5.0)


for p in itertools.islice(amzn.search(Keywords='python', SearchIndex='Books'), 5):
    print(p.title)