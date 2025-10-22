# src/marketing_posts/crew.py
import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field
from typing import List

# --- Tool Imports ---
# We keep these for research
from crewai_tools import ScrapeWebsiteTool, TavilySearchTool 
# We add our new custom tools
from marketing_posts.tools.shopify_tools import LoadStoreProductsTool, UpdateProductSEOTool

# --- LLM Imports ---
from langchain_ollama import Ollama as OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# --- Initialize LLMs (Same as before, but with OllamaLLM) ---
search_tool = TavilySearchTool()
scrape_tool = ScrapeWebsiteTool()
load_products_tool = LoadStoreProductsTool()
update_seo_tool = UpdateProductSEOTool()

ollama_llm = OllamaLLM(...) # Your Ollama setup
gemini_fallback = ...      # Your Gemini setup
openai_fallback = ...      # Your OpenAI setup

# --- Pydantic Models (Can be the same or modified) ---
class Copy(BaseModel):
    title: str = Field(..., description="Title of the copy")
    body: str = Field(..., description="Body of the copy")

# A new model for SEO
class ProductSEO(BaseModel):
    product_id: int = Field(..., description="The ID of the product to update")
    seo_title: str = Field(..., description="The new, optimized SEO title (max 60 chars)")
    seo_description: str = Field(..., description="The new, optimized SEO description (max 160 chars)")

# --------------------------------------------------------------------
# 3. DEFINE YOUR NEW CREW
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
            config=self.agents_config['shopify_analyst'], # You'll need to update agents.yaml
            tools=[load_products_tool, search_tool], # Can load products and search
            verbose=True
        )

    @agent
    def seo_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['seo_specialist'], # You'll need to update agents.yaml
            tools=[search_tool, scrape_tool, update_seo_tool], # Can research and update SEO
            verbose=True
        )

    @agent
    def marketing_copywriter(self) -> Agent:
        return Agent(
            config=self.agents_config['creative_content_creator'], # Can reuse the old one
            verbose=True
            # No tools needed, just writes based on context
        )

    # --- NEW TASKS ---

    @task
    def load_products_task(self) -> Task:
        return Task(
            config=self.tasks_config['load_products_task'], # New task in tasks.yaml
            agent=self.shopify_analyst()
            # This task's output will be the list of all products
        )

    @task
    def seo_optimization_task(self) -> Task:
        return Task(
            config=self.tasks_config['seo_optimization_task'], # New task in tasks.yaml
            agent=self.seo_specialist(),
            context=[self.load_products_task()], # Depends on products being loaded
            output_json=ProductSEO 
            # We want structured output for *each* product
            # Note: You may need to adjust this to output a List[ProductSEO]
        )
    
    @task
    def marketing_campaign_task(self) -> Task:
        return Task(
            config=self.tasks_config['marketing_campaign_task'], # New task in tasks.yaml
            agent=self.marketing_copywriter(),
            context=[self.seo_optimization_task()], # Runs after SEO is done
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
            fallback_llm=[gemini_fallback, openai_fallback]
        )