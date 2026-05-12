from openai import OpenAI

client = OpenAI(
    api_key='sk-**',  # 替换为您的令牌
    base_url='https://yeysai.com/v1'
)


messages = [
    {
        "role": "system",
        "content": "你是一个专业的AI助手，能够帮助用户解决各种问题。"
    },
    {
        "role": "user",
        "content": "请帮我设计一个1*16单元的天线阵列，10GHz中心频点"
    }
]


response = client.chat.completions.create(
    messages=messages,
    model="gpt-4o"  # 可选模型：gpt-4o, gpt-4o-mini, claude-3-5-sonnet-20241022, llama-3.2-3b-preview 等
).model_dump()


print(response['choices'][0]['message']['content'])
print(response)