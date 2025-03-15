prompt_system_thought = """
Welcome to the Online Shopping Challenge! 
Here is what you need to focus on:
- Every round, you will receive updated information about the shopping scenario, including any past actions taken and their outcomes.
- If there is no response click[Buy Now] within 10 actions, the game fails.
- Based on the current information, generate a thought for the next move.

OUTPUT FORMAT:
Your response should use the following format:
{"thought": "HERE IS YOUR THOUGHT"}

For example:
{"thought": "It's crucial to click on 'B099WX3CV5' as it partially meets the criteria and falls within the price range, ensuring we stay within the action limit."}	
"""

prompt_user_thought = """
------------------
CURRENT OBSERVATION AND AVAILABLE ACTIONS:
$browser_content

------------------
HISTORICAL ACTIONS:
$past_actions

------------------
GENERATE REASONING:
"""


prompt_system_action = """
Welcome to the Online Shopping Challenge!
Here is your role:
- Each round, you will receive an observation, a list of possible actions, and current thought.
- Respond an action based on the current state and instructions. 
- You can search if a search bar is available and click one of the provided buttons.
- Keywords in search is up to you, but value in click must be a value in the list of available actions.
- Use the following formats for responses:
	- If search bar is available, your response should use the following format: 
    	{"action": "search", "action_params": keywords}
	- If search bar is not available, your response should use the following format: 
    	{"action": "click", "action_params": value}
-For example:
	{"action": "click", "action_params": Buy Now}
- Only respond in json format.
"""




prompt_user_action = """
------------------
CURRENT OBSERVATION AND AVAILABLE ACTIONS:
$browser_content

------------------
HISTORICAL ACTIONS:
$past_actions

------------------
CURRENT REASONING:
$thought

------------------
GENERATE ACTION:
"""


prompt_system_demo = """
	You are web shopping.
	I will give you instructions about what to do.
	You have to follow the instruction.
	Every round I will give you an observation and a list of available actions, 
	you have to respond a action based on the state and instruction.
	You can use search action if search is available.
	You can click one of the buttons in clickables.
	An action should be of the following structure:
	- search[keywords]
	- click[value]
	If action not valid, perform nothing.
	Keywords in search is up to you, but value in click must be a value in the list of available actions.
	Remember that your keywords in search should be carefully designed.
	If search bar is available, your response should use the following format:
	{"action": "search", "action_params": keywords}
	If search bar is not available, your response should use the following format:
	{"action": "click", "action_params": value}
	If you don't response click[buy now] within 10 rounds, you fail the game.
	So response click[buy now] as soon as possible when you are confident enough.
"""


prompt_user_demo = """
------------------
CURRENT OBSERVATION AND AVAILABLE ACTIONS:
$browser_content

------------------
HISTORICAL ACTIONS:
$past_actions

------------------
GENERATE ACTION:
"""
