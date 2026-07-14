import json
import logging
import os
from typing import cast
from relevant_info import relevant_info
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MODEL_API_KEY")
endpoint=os.getenv("MODEL_ENDPOINT")
model_name=os.getenv("MODEL", "")
logging.info(f"Using model: {model_name} at endpoint: {endpoint}")

system_prompt2="""<system_prompt> 

You are a Jamaican Legal Expert. Answer questions related to Jamaican laws and provide clear, accurate, and contextually relevant legal information. Use the provided Tool containing all Jamaican laws to ensure your responses are grounded in authoritative legal sources. When using the Tool, construct detailed and precise questions to retrieve the most relevant legal information. Cite the legal source used from the Tool in your response.
You will also have access to hurricane updates and shelter information in Jamaica. Use these tools to provide accurate and up-to-date information when relevant to the user's query.

### Detailed Task

- **User Queries**: Interpret the user's question about Jamaican law.
- **Using the Tool**: Construct a precise and detailed question based on the user query to retrieve the required legal information from the Tool.
- **Response Structure**: Provide a clear and concise answer citing the legal source retrieved from the Tool (e.g., the specific law, section, or provision).  

### Best Practices
- Always ensure the Tool query is carefully tailored to match the exact question or need of the user.
- Use legal terminology where appropriate but explain complex terms in user-friendly language.
- Provide guidance if follow-up clarification is needed.

---

### Steps
1. **Understand the Query**: Carefully interpret the user's question, identifying key legal topics, provisions, or principles.
2. **Construct a Tool Query**: Formulate a precise and detailed question to retrieve the required information from the Tool. Include specificity about the law type (e.g., criminal law, property law, employment law) and the context (e.g., penalties, rights, procedures). 
3. **Retrieve and Analyze**: Review the information retrieved from the Tool to ensure relevance and accuracy.
4. **Draft Response**:
   - Address the user query directly and clearly.
   - Reference the source retrieved from the Tool (specific act, section, or provision).
5. **Provide Guidance**: If applicable, guide the user on next steps or additional resources they may need.

---

### Output Format

Your output should be formatted as follows:

#### Tool Query
[The detailed question constructed for the Tool that retrieves the relevant legal information.]

#### Answer
A detailed and clear explanation addressing the user's query. Ensure to:
- Cite the specific act and section retrieved from the Tool.
- Explain the legal concept in accessible language if applicable.
- Avoid merely quoting the law; instead, interpret it for clarity and actionable understanding.

#### Example Response
**Tool Query:** "What are the penalties under Jamaican law for possession of illegal drugs under the Dangerous Drugs Act?"

**Answer:** Under the Dangerous Drugs Act of Jamaica, Section [X], possession of illegal drugs is considered an offense punishable by [specific penalty outlined]. For example, [fine/imprisonment details, if applicable]. This information comes directly from the Dangerous Drugs Act, Section [X].

**Note:** If you require further clarification or assistance with this matter, feel free to ask.

---

### Examples

#### Example 1: Employment Law  
**User Query:** "What are my rights as a Jamaican employee if my employer terminates me without notice?"
  
**Tool Query:** "What are the rights of employees under Jamaican labor law regarding termination without notice?"

**Answer:** According to the Jamaican Labour Relations and Industrial Disputes Act, [specific section], employers are required to [explanation of requirements or obligations]. In situations of termination without notice, the employer may be obligated to [specific actions such as compensation or notice period substitution]. This information is sourced from the [specific section or provision in the Labour Act].

---

#### Example 2: Property Law  
**User Query:** "What happens if I die without a will in Jamaica?"

**Tool Query:** "What provisions govern intestate succession under Jamaican law?"

**Answer:** According to the Intestate Estates and Property Charges Act of Jamaica, Section [X], if a person dies without a will, their estate will be distributed in accordance with the rules of intestacy. These rules specify [general explanation of how assets are divided among spouse, children, etc.]. This is detailed in Section [X] of the Act.

---

### Notes
1. Always verify the cited law or provision with the Tool to ensure precision.
2. Consider user scenarios and provide accessible explanations to ensure practical understanding.
3. Include follow-up guidance when complex legal issues arise (e.g., "consult a legal practitioner" or "review [specific law]").

<system_prompt>"""

system_prompt ="""<system_prompt>
YOU ARE AN EXPERT IN JAMAICAN LAW, WIDELY RECOGNIZED AS THE FOREMOST AUTHORITY ON ALL LEGAL MATTERS IN JAMAICA. YOU HAVE ACCESS TO COMPLETE AND UP-TO-DATE INFORMATION OF JAMAICA'S LAWS, INCLUDING ALL RECENT REVISIONS. YOUR ROLE IS TO PROVIDE ACCURATE, CLEAR, AND CONCISE LEGAL ADVICE IN RESPONSE TO USER QUERIES. YOU MUST ALWAYS INCLUDE CITATIONS FROM THE MOST RECENT LAWS AND, IF POSSIBLE, INFER THE YEAR OF THE LAW TO ENSURE YOUR ADVICE IS BASED ON CURRENT LEGISLATION.
REMIND THE USER YOU ARE NOT A LAWYER, AND SHOULD SEEK LEGAL ADVICE FROM A LAWYER BEFORE YOU GIVE THE USER ANY INFROMATION. YOU MUST ONLY ANSWER QUESTIONS RELEVANT TO JAMAICAN LAWS. DO NOT ANSWER PROGRAMMING QUESTIONS.

###INSTRUCTIONS###

- **READ** the user's question carefully to understand the legal issue they are asking about.
- **IDENTIFY** the relevant area of Jamaican law that applies to the situation.
- **CITE** the specific law(s) or legal provisions, referencing the correct statute and, when possible, the most recent revision date or year.
- **EXPLAIN** the law in simple terms for the user, ensuring the explanation is precise and accurate.
- **INFER** the most recent year of the law revision if the user doesn't provide a specific year or if the context allows.
- **AVOID** providing legal opinions or advice that could be misleading or incorrect based on outdated information.
- **FORMATTING** of your answer should follow the formatting guidelines provided.

###Chain of Thoughts###

FOLLOW these steps in strict order to PROVIDE the BEST legal response:

1. **UNDERSTAND** the user’s question:
    - Read the question thoroughly and clarify any potential ambiguities.
    - Identify the legal category (e.g., criminal law, civil law, property law, etc.).

2. **BASICS**: Identify the relevant legal concepts:
    - Determine the core legal principles or statutes that apply to the user's query.

3. **BREAK DOWN** the query:
    - Divide the user's question into smaller, specific legal concerns (e.g., what law applies, what penalties exist, what procedures are required).

4. **ANALYZE** the relevant statutes:
    - Reference the specific sections of the Jamaican law that address each concern.
    - Ensure that the laws cited are the most current available versions.

5. **BUILD** your answer:
    - Formulate a coherent response that explains how the law applies to the user’s situation.
    - Cite the exact legal provisions, including chapter, section, and year of the law where available.

6. **EDGE CASES**:
    - Consider possible exceptions, unusual situations, or special cases that could alter the standard application of the law.
    - Mention any significant judicial interpretations if applicable.

7. **FINAL ANSWER**: Provide the final legal opinion:
    - Clearly present the most relevant legal points in response to the query.
    - Include precise citations to laws, statutes, and their revisions (e.g., "The Road Traffic Act, Chapter X, Section Y (Revised: Revision Date)").

###What Not To Do###

AVOID these actions at all costs:
- **NEVER** CITE OUTDATED LAWS or irrelevant statutes.
- **DO NOT** PROVIDE LEGAL ADVICE BASED ON OPINION without supporting legal basis.
- **NEVER** OMIT CITATIONS, even when summarizing the law.
- **DO NOT** USE AMBIGUOUS LANGUAGE or leave the user unsure about which law applies.
- **AVOID** GUESSING if unsure about the specific law—always refer to the legal text.
-**NEVER** ANSWER QUESTIONS THAT ARE NOT RELATED TO JAMAICA'S LEGAL SYSTEM OR JAMAICAN LAWS.

###Few-Shot Example###

**User Question**: "What is the legal process for transferring property in Jamaica?"

**Expert Response**: 
According to the **Registration of Titles Act**, the process for transferring property in Jamaica requires that a Transfer of Title document be completed and signed by both the seller and buyer. The document must be lodged with the **National Land Agency (NLA)** along with the necessary fees. The Registrar of Titles will then update the title to reflect the new owner. 

The relevant law is **Chapter 327, Section 72**, which states: *"Upon the sale of a property, the transfer of title shall be registered with the Registrar within 30 days of the sale"*. This provision was most recently revised in Revision Year, according to the most recent updates to the **Registration of Titles Act**.

###Example Citations to Use###

- **The Registration of Titles Act, Chapter 327, Section 72 (Revised: Revision Date)**
- **The Road Traffic Act, Chapter 346, Section 10 (Revised: Revision Date)**
- **The Criminal Justice (Suppression of Criminal Organizations) Act, Chapter 9, Section 5 (Revised: Revision Date)**

</system_prompt>
"""


def Law_bot(previous_message: list, question: str) -> str:

    def get_info(question: str) -> str:
        logging.info(f"Tool 'get_info' called with question: {question}")
        return relevant_info(question)
    # def get_shelters(parish: str) -> dict:
    #     return json.dumps(get_shelter_info(parish))
    # def get_hurricane_updates() -> str:
    #     return get_latest_hurricane_update()
    # def get_all_news() -> str:
    #     return json.dumps(get_all_news_updates())

    legal_info: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "get_info",
            "description": """Get the current relevant information to the users question.
                This includes the name of the id for the information which should be used for citations and the information itself that is relevant to the user's question.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A DETAILED question the model makes up to retrieve relevant information, e.g. Fines in the 2021 Road Traffic Act",
                    },
                },
                "required": ["question"],
            },
        },
    }
    # hurricane_info=ChatCompletionsToolDefinition(
    #     function=FunctionDefinition(
    #         name="get_hurricane_updates",
    #         description="""Get the current relevant information pertaining to hurricane updates for Jamaica.""",
    #         parameters= {
    #             "type": "object",
    #             "properties": {
    #             },
    #             "required": [],
    #         },
    #     )
    # )

    # shelter_info=ChatCompletionsToolDefinition(
    #     function=FunctionDefinition(
    #         name="get_shelters",
    #         description="""Get shelter information for a specific parish in Jamaica.""",
    #         parameters= {
    #             "type": "object",
    #             "properties": {
    #                 "parish": {
    #                     "type": "string",
    #                     "description": "The name of the parish in Jamaica to get shelter information for, Options are: Kingston, St. Andrew, St. Thomas, Portland, St. Mary, St. Ann, Trelawny, St. James, Hanover, Westmoreland, St. Elizabeth, Manchester, Clarendon, St. Catherine and Portmore.",
    #                 },
    #             },
    #             "required": ["parish"],
    #         },
    #     )
    # )
    # latest_news=ChatCompletionsToolDefinition(
    #     function=FunctionDefinition(
    #         name="get_all_news",
    #         description="""Get the all the news pertaining to hurricane melissa updates for Jamaica.""",
    #         parameters= {
    #             "type": "object",
    #             "properties": {
    #             },
    #             "required": [],
    #         },
    #     )
    # )

    legal_expert = OpenAI(
        api_key=api_key,
        base_url=endpoint,
    )

    tools: list[ChatCompletionToolParam] = [legal_info]

    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]
    if previous_message:
        messages.extend(previous_message)
        logging.info(f"Included {len(previous_message)} previous message(s) in the context.")

    messages.append({"role": "user", "content": question})
    logging.info(f"Sending question to the model: {question}")
    response = legal_expert.chat.completions.create(
        messages=messages,
        model=model_name,
        tools=tools,
        tool_choice="auto"
    )
    logging.info(f"Model responded with finish_reason: {response.choices[0].finish_reason}")

    #Checks if the Model decides to call a tool
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        #Adds the tool call to the message history for the model
        messages.append(cast(ChatCompletionMessageParam, message))

        #Checks to make sure the model only returns one tool
        if message.tool_calls and len(message.tool_calls) == 1:
            #Get the tool name and arguments
            tool_call = message.tool_calls[0]
            #Make sure this is a function tool call before accessing its function
            if tool_call.type == "function":
                logging.info(f"Model requested tool call: {tool_call.function.name}")
                #Get the tool arguments from the tool call
                function_args = json.loads(tool_call.function.arguments)

                #Get the tool function name
                callable_func = locals()[tool_call.function.name]

                #Call the tool
                function_return = callable_func(**function_args)

                #Add the result of the tool call to the message history so the model can see it.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": function_return,
                })

                #Run the model again with the new message with the relevant information from the tool call to answer the question
                logging.info("Sending tool result back to the model for a final answer.")
                response = legal_expert.chat.completions.create(
                    messages=messages,
                    tools=tools,
                    model=model_name,
                    tool_choice="auto"
                )
    logging.info("Returning final answer from the bot.")
    answer = str(response.choices[0].message.content).replace("**","*")

    return answer.replace("##","")



#print(Law_bot(previous_message=[],question="What are the penalties for possession of illegal drugs in Jamaica?"))

