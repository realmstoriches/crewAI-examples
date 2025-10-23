import os
import shopify
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# --- Tool Setup ---

def _setup_shopify_session():
    """Initializes the Shopify API session from .env variables."""
    shop_url = os.getenv("SHOPIFY_SHOP_URL")
    api_version = os.getenv("SHOPIFY_API_VERSION")
    access_token = os.getenv("SHOPIFY_ADMIN_ACCESS_TOKEN")

    if not all([shop_url, api_version, access_token]):
        raise ValueError("Shopify credentials (SHOPIFY_SHOP_URL, SHOPIFY_API_VERSION, SHOPIFY_ADMIN_ACCESS_TOKEN) not set in .env file.")
        
    session = shopify.Session(shop_url, api_version, access_token)
    shopify.ShopifyResource.activate_session(session)

def _get_all_products():
    """Helper to fetch all products, handling pagination."""
    _setup_shopify_session() # Ensure session is active
    
    products = []
    try:
        page = shopify.Product.find()
        products.extend(page)
        while page.has_next_page():
            page = page.next_page()
            products.extend(page)
    except Exception as e:
        return f"Error fetching products: {str(e)}"

    # We only care about a few fields to save tokens
    simplified_products = [
        {
            "id": p.id,
            "title": p.title,
            "handle": p.handle,
            "body_html": p.body_html, # The description
            "tags": p.tags,
            "seo_title": p.metafields_global_title_tag,
            "seo_description": p.metafields_global_description_tag,
        }
        for p in products
    ]
    return simplified_products

def _update_product_seo(product_id: int, seo_title: str, seo_description: str):
    """Helper to update a single product's SEO."""
    _setup_shopify_session() # Ensure session is active
    
    try:
        product = shopify.Product.find(product_id)
        product.metafields_global_title_tag = seo_title
        product.metafields_global_description_tag = seo_description
        
        if product.save():
            return f"Successfully updated SEO for product ID {product_id}."
        else:
            return f"Failed to update SEO for product ID {product_id}. Errors: {product.errors.full_messages()}"
    except Exception as e:
        return f"Error updating product {product_id}: {str(e)}"

# --- Tool Argument Schemas ---
# This makes the tools more reliable

class UpdateSEOSchema(BaseModel):
    product_id: int = Field(..., description="The ID of the product to update")
    seo_title: str = Field(..., description="The new, optimized SEO title (max 70 chars)")
    seo_description: str = Field(..., description="The new, optimized SEO description (max 160 chars)")

# --- Tool Definitions ---

class LoadStoreProductsTool(BaseTool):
    name: str = "Load All Store Products"
    description: str = "Loads all products from the Shopify store, returning a simplified JSON list with product ID, title, body, tags, and current SEO fields."
    
    def _run(self) -> str:
        return _get_all_products()

class UpdateProductSEOTool(BaseTool):
    name: str = "Update Product SEO"
    description: str = "Updates the SEO meta title and meta description for a specific product using its product_id."
    args_schema: BaseModel = UpdateSEOSchema
    
    def _run(self, product_id: int, seo_title: str, seo_description: str) -> str:
        return _update_product_seo(product_id, seo_title, seo_description)