from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets
import output_parsers


def ice_breaker(name: str) -> str:
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    # twitter_username = twitter_lookup_agent(name=name)
    # tweets = scrape_user_tweets(username=twitter_username, num_tweets=5)

    summary_template = """
         given the Linkedin information {linkedin_information} and about a person from I want you to create:
         1. a short summary
         2. two interesting facts about them
         3. A topic that may interest them
         4. 2 creative Ice breakers to open a conversation with them 
        \n{format_instructions}
     """

    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_information"],
        template=summary_template,
        partial_variables = {"format_instructions":output_parsers.person_intel_parser.get_format_instructions()}
    )

    llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    result = chain.run(linkedin_information=linkedin_data)
    print(result)
    return result


if __name__ == "__main__":
    print("Hello LangChain!")
    ice_breaker(name="Paolo Zaccardi Fabrick")
