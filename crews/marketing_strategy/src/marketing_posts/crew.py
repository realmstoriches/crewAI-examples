import os
from typing import List
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field

# --- Tool Imports ---
from crewai_tools import ScrapeWebsiteTool, TavilySearchTool
# --- Our New Shopify Tools ---
from marketing_posts.tools.shopify_tools import LoadStoreProductsTool, UpdateProductSEOTool

# --- LLM Imports ---
from langchain_ollama import OllamaLLM             # <-- UPDATED
# from langchain_google_genai import ChatGoogleGenerativeAI # <-- REMOVED
from langchain_openai import ChatOpenAI

# --------------------------------------------------------------------
# 1. DEFINE YOUR TOOLS AND LLMS
# --------------------------------------------------------------------

# --- Search Tools ---
search_tool = TavilySearchTool()
scrape_tool = ScrapeWebsiteTool()

# --- Shopify Tools ---
load_products_tool = LoadStoreProductsTool()
update_seo_tool = UpdateProductSEOTool()


# --- LLMs ---
# 1. Main LLM: Ollama
ollama_llm = OllamaLLM( # <-- UPDATED
    model=os.getenv("OLLAMA_MODEL", "llama3"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

# 2. Fallback LLMs
openai_fallback = ChatOpenAI(model="gpt-4o-mini")
# gemini_fallback = ... # <-- REMOVED

# --------------------------------------------------------------------
# 2. DEFINE YOUR DATA MODELS (Pydantic)
# --------------------------------------------------------------------

# This model is for our new SEO task
class ProductSEO(BaseModel):
    """Product SEO Optimization model"""
    product_id: int = Field(..., description="The ID of the product to update")
    seo_title: str = Field(..., description="The new, optimized SEO title (max 70 chars)")
    seo_description: str = Field(..., description="The new, optimized SEO description (max 160 chars)")

# This model is for our final marketing copy task
class Copy(BaseModel):
    """Copy model"""
    title: str = Field(..., description="Title of the marketing copy/campaign")
    body: str = Field(..., description="Body of the marketing copy")

# --------------------------------------------------------------------
# 3. DEFINE YOUR NEW SHOPIFY CREW
# --------------------------------------------------------------------

@CrewBase
class MarketingPostsCrew():
    """Shopify Marketing & SEO Crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # --- NEW AGENTS ---
    
    @agent
    def shopify_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['shopify_analyst'],
            tools=[load_products_tool, search_tool, scrape_tool],
            verbose=True
        )

    @agent
    def seo_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['seo_specialist'],
            tools=[search_tool, scrape_tool, update_seo_tool], # Can update SEO
            verbose=True
        )

    @agent
    def marketing_copywriter(self) -> Agent:
        return Agent(
            config=self.agents_config['marketing_copywriter'],
            tools=[search_tool, scrape_tool],
            verbose=True
        )

    # --- NEW TASKS ---

    @task
    def load_products_task(self) -> Task:
        return Task(
            config=self.tasks_config['load_products_task'],
            agent=self.shopify_analyst()
            # This task's output will be the list of all products
        )

    @task
    def seo_optimization_task(self) -> Task:
        return Task(
            config=self.tasks_config['seo_optimization_task'],
            agent=self.seo_specialist(),
            context=[self.load_products_task()], # Depends on products being loaded
            output_json=ProductSEO 
            # We want structured output for *each* product
            # Note: This will ask the agent to apply changes to ALL products.
        )
    
    @task
    def marketing_campaign_task(self) -> Task:
        return Task(
            config=self.tasks_config['marketing_campaign_task'],
            agent=self.marketing_copywriter(),
            context=[self.load_products_task(), self.seo_optimization_task()], # Has all info
            output_json=Copy
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Shopify SEO crew"""
        return Crew(
            agents=[self.shopify_analyst(), self.seo_specialist(), self.marketing_copywriter()],
            tasks=[self.load_products_task(), self.seo_optimization_task(), self.marketing_campaign_task()],
            process=Process.sequential,
            verbose=True,
            llm=ollama_llm,
            # Only use OpenA-I as a fallback
            fallback_llm=[openai_fallback] # <-- UPDATED
        )