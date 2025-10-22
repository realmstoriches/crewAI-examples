import os
from typing import List
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field

# --- Tool Imports ---
# We've replaced SerperDevTool with TavilySearchResults
from crewai_tools import ScrapeWebsiteTool, TavilySearchResults

# --- LLM Imports ---
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# --------------------------------------------------------------------
# 1. DEFINE YOUR TOOLS AND LLMS
# --------------------------------------------------------------------

# (Tools will read TAVILY_API_KEY from .env)
search_tool = TavilySearchResults()
scrape_tool = ScrapeWebsiteTool()

# (LLMs will read their keys from .env)

# 1. Main LLM: Ollama
# It reads the URL and model from your .env file
# (Defaults to llama3 and localhost:11434 if not set)
ollama_llm = Ollama(
    model=os.getenv("OLLAMA_MODEL", "llama3"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

# 2. Fallback LLMs
# These will automatically use GEMINI_API_KEY and OPENAI_API_KEY
gemini_fallback = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
openai_fallback = ChatOpenAI(model="gpt-4o-mini")

# --------------------------------------------------------------------
# 2. DEFINE YOUR DATA MODELS (Pydantic)
# (This section is unchanged)
# --------------------------------------------------------------------

class MarketStrategy(BaseModel):
	"""Market strategy model"""
	name: str = Field(..., description="Name of the market strategy")
	tatics: List[str] = Field(..., description="List of tactics to be used in the market strategy")
	channels: List[str] = Field(..., description="List of channels to be used in the market strategy")
	KPIs: List[str] = Field(..., description="List of KPIs to be used in the market strategy")

class CampaignIdea(BaseModel):
	"""Campaign idea model"""
	name: str = Field(..., description="Name of the campaign idea")
	description: str = Field(..., description="Description of the campaign idea")
	audience: str = Field(..., description="Audience of the campaign idea")
	channel: str = Field(..., description="Channel of the campaign idea")

class Copy(BaseModel):
	"""Copy model"""
	title: str = Field(..., description="Title of the copy")
	body: str = Field(..., description="Body of the copy")

# --------------------------------------------------------------------
# 3. DEFINE YOUR CREW
# (This is where all the changes are applied)
# --------------------------------------------------------------------

@CrewBase
class MarketingPostsCrew():
	"""MarketingPosts crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def lead_market_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['lead_market_analyst'],
			# --- MODIFIED ---
			tools=[search_tool, scrape_tool],  # Replaced Serper with Tavily
			llm=ollama_llm,                      # Set main LLM
			fallback_llm=[gemini_fallback, openai_fallback], # Set fallbacks
			# --- END MODIFIED ---
			verbose=True,
			memory=False,
		)

	@agent
	def chief_marketing_strategist(self) -> Agent:
		return Agent(
			config=self.agents_config['chief_marketing_strategist'],
			# --- MODIFIED ---
			tools=[search_tool, scrape_tool],  # Replaced Serper with Tavily
			llm=ollama_llm,                      # Set main LLM
			fallback_llm=[gemini_fallback, openai_fallback], # Set fallbacks
			# --- END MODIFIED ---
			verbose=True,
			memory=False,
		)

	@agent
	def creative_content_creator(self) -> Agent:
		return Agent(
			config=self.agents_config['creative_content_creator'],
			# --- MODIFIED ---
			# This agent doesn't need tools, but it needs the LLMs
			llm=ollama_llm,                      # Set main LLM
			fallback_llm=[gemini_fallback, openai_fallback], # Set fallbacks
			# --- END MODIFIED ---
			verbose=True,
			memory=False,
		)

	# --- TASKS (Unchanged) ---
 
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			agent=self.lead_market_analyst()
		)

	@task
	def project_understanding_task(self) -> Task:
		return Task(
			config=self.tasks_config['project_understanding_task'],
			agent=self.chief_marketing_strategist()
		)

	@task
	def marketing_strategy_task(self) -> Task:
		return Task(
			config=self.tasks_config['marketing_strategy_task'],
			agent=self.chief_marketing_strategist(),
			output_json=MarketStrategy
		)

	@task
	def campaign_idea_task(self) -> Task:
		return Task(
			config=self.tasks_config['campaign_idea_task'],
			agent=self.creative_content_creator(),
   			output_json=CampaignIdea
		)

	@task
	def copy_creation_task(self) -> Task:
		return Task(
			config=self.tasks_config['copy_creation_task'],
			agent=self.creative_content_creator(),
   			context=[self.marketing_strategy_task(), self.campaign_idea_task()],
			output_json=Copy
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the MarketingPosts crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)