#!/usr/bin/env python
import sys
from dotenv import load_dotenv
load_dotenv() # <-- This MUST be at the top

from marketing_posts.crew import MarketingPostsCrew

def run():
    # --- NEW INPUTS ---
    # These will be injected into your task descriptions
    inputs = {
        'target_audience': 'Tech-savvy professionals and small business owners who value high-quality, unique products.',
        'brand_voice': 'Professional, slightly playful, innovative, and customer-focused.',
        'product_type': 'Unique 3D printed gadgets and home decor' # Change this to your store's general category
    }
    MarketingPostsCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'target_audience': 'Tech-savvy professionals and small business owners who value high-quality, unique products.',
        'brand_voice': 'Professional, slightly playful, innovative, and customer-focused.',
        'product_type': 'Unique 3D printed gadgets and home decor' # Change this to your store's general category
    }
    try:
        MarketingPostsCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
    
    
if __name__ == "__main__":
    run() if len(sys.argv) <= 1 else train()