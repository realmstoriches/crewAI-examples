# src/marketing_posts/tools/shopify_tools.py
import os
import shopify
from crewai_tools import BaseTool

class ShopifyTools:
    def __init__(self):
        self._setup_shopify_session()

    def _setup_shopify_session(self):
        """Initializes the Shopify API session."""
        shop_url = os.getenv("SHOPIFY_SHOP_URL")
        api_version = os.getenv("SHOPIFY_API_VERSION")
        access_token = os.getenv("SHOPIFY_ADMIN_ACCESS_TOKEN")

        if not all([shop_url, api_version, access_token]):
            raise ValueError("Shopify credentials not set in .env file.")
            
        session = shopify.Session(shop_url, api_version, access_token)
        shopify.ShopifyResource.activate_session(session)

    def _get_all_products(self):
        """Helper to fetch all products, handling pagination."""
        products = []
        page = shopify.Product.find()
        while page:
            products.extend(page)
            if not page.has_next_page():
                break
            page = page.next_page()
        
        # We only care about a few fields to save tokens
        simplified_products = [
            {
                "id": p.id,
                "title": p.title,
                "handle": p.handle,
                "body_html": p.body_html,
                "tags": p.tags,
                "seo_title": p.metafields_global_title_tag,
                "seo_description": p.metafields_global_description_tag,
            }
            for p in products
        ]
        return simplified_products

    def _update_product_seo(self, product_id, seo_title, seo_description):
        """Helper to update a single product's SEO."""
        try:
            product = shopify.Product.find(product_id)
            product.metafields_global_title_tag = seo_title
            product.metafields_global_description_tag = seo_description
            
            if product.save():
                return f"Successfully updated SEO for product ID {product_id}."
            else:
                return f"Failed to update SEO for product ID {product_id}."
        except Exception as e:
            return f"Error updating product {product_id}: {str(e)}"

# --- Tool Definitions ---

class LoadStoreProductsTool(BaseTool):
    name: str = "Load All Store Products"
    description: str = "Loads all products from the Shopify store, returning a simplified JSON list with product ID, title, body, tags, and current SEO fields."
    
    def _run(self) -> str:
        tools = ShopifyTools()
        return tools._get_all_products()

class UpdateProductSEOTool(BaseTool):
    name: str = "Update Product SEO"
    description: str = "Updates the SEO meta title and meta description for a specific product using its product_id."
    
    def _run(self, product_id: int, seo_title: str, seo_description: str) -> str:
        tools = ShopifyTools()
        return tools._update_product_seo(product_id, seo_title, seo_description)