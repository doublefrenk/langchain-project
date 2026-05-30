from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
import os

load_dotenv()

MAX_ITERATION=10

model = os.getenv("LLM_MODEL")

@tool
def get_product_price(product_name: str) -> str:
    """Given a product name, return its price."""
    prices = {
        "Laptop": "$799",
        "MacBook Pro": "$1299",
        "AirPods Pro": "$249",
    }
    return prices.get(product_name, "Product not found")

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """ Aplly discount to a price based on the discount tier.
    Discount tiers: bronze (5%), silver (10%), gold (15%) """
    discounts = {
        "bronze": 0.05,
        "silver": 0.10,
        "gold": 0.15,
    }

    discount_percentage = discounts.get(discount_tier, 0)
    return round(price * (1 - discount_percentage), 2)

# Agent loop

llm = init_chat_model(model=model, temperature=0)
llm_with_tools = llm.bind_tools([get_product_price, apply_discount])

@traceable(name="agent_loop", tags=["agent", "loop"])
def run_agent(question: str) :
    tools = [get_product_price, apply_discount]
    tool_dict = {t.name: t for t in tools}

    messages = [
    SystemMessage(
              content=(
                  "You are a helpful shopping assistant. "
                  "You have access to a product catalog tool "
                  "and a discount tool.\n\n"
                  "STRICT RULES — you must follow these exactly:\n"
                  "1. NEVER guess or assume any product price. "
                  "You MUST call get_product_price first to get the real price.\n"
                  "2. Only call apply_discount AFTER you have received "
                  "a price from get_product_price. Pass the exact price "
                  "returned by get_product_price — do NOT pass a made-up number.\n"
                  "3. NEVER calculate discounts yourself using math. "
                  "Always use the apply_discount tool.\n"
                  "4. If the user does not specify a discount tier, "
                  "ask them which tier to use — do NOT assume one."
              )
          ),
          HumanMessage(content=question),]

    for iteration in range(1, MAX_ITERATION +1):
      print(f"--- Iteration {iteration} ---")

      ai_message = llm_with_tools.invoke(messages)

      # La risposta dell'agente potrebbe essere una normale risposta testuale, oppure una chiamata a uno strumento. Se è una chiamata a uno strumento, sarà rappresentata come un ToolMessage, che include il nome dello strumento e i suoi argomenti.

      tool_calls = ai_message.tool_calls

      if  not tool_calls:
          print("Agent response:", ai_message.content)
          return ai_message.content

      for tool_call in tool_calls:
          tool_name = tool_call.get("name")
          tool_args = tool_call.get("args", {})
          tool_call_id = tool_call.get("id")

          print(f"[Tool selected]: {tool_name} with args {tool_args}")

          tool_to_use = tool_dict.get(tool_name)

          if tool_to_use is None:
              raise ValueError(f"Tool {tool_name} not found")

          observation = tool_to_use.invoke(tool_args)

          print(f"[Tool result]: {observation}")

          messages.append(ai_message)
          messages.append(ToolMessage(content=observation, tool_call_id=tool_call_id))

    print("Error: Maximum iterations reached without a final answer.")
    return None



if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    result = run_agent("What is the price of a Laptop and apply a gold discount?")

