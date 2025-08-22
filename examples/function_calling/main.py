from openai import OpenAI
import json

client = OpenAI(
    api_key='1c8161ec90ddc3da12b12adc2160f0e7',
    base_url='https://llm-proxy.seznam.net/v1'
)

# 1. Define the function (tool)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_horoscope",
            "description": "Get today's horoscope for an astrological sign.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sign": {
                        "type": "string",
                        "description": "An astrological sign like Taurus or Aquarius",
                    },
                },
                "required": ["sign"],
            },
        },
    }
]

# 2. Create initial input messages
messages = [
    {"role": "user", "content": "What is my horoscope? I am an Aquarius."}
]

# 3. Send to OpenAI with tools
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

# 4. Extract tool call
tool_call = response.choices[0].message.tool_calls[0]
print(f"Tool call: {tool_call}")
function_name = tool_call.function.name
print(f"Function name: {function_name}")
arguments = json.loads(tool_call.function.arguments)
print(f"Arguments: {arguments}")

# 5. Call local function
def get_horoscope(sign):
    return f"{sign}: Next Tuesday you will befriend a baby otter."

function_result = get_horoscope(arguments["sign"])

# 6. Add tool call + function response to conversation
messages.append({
    "role": "assistant",
    "tool_calls": [tool_call]
})
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "name": function_name,
    "content": function_result,
})

# 7. Final response
final_response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
)

print(final_response.choices[0].message.content)
