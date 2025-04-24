from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch, RunnableParallel
from operator import itemgetter
import vertexai
import json
import os
import dotenv
import asyncio
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
#  this is for anthropic claude


dotenv.load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

class Chain:

    def __init__(self):
        self.llm = None

    async def initialize(self):
        vertexai.init(project=os.getenv("PROJECT"), location=os.getenv("LOCATION_1"))
        self.llm = await self.initialize_llm()

    '''def __init__(self):
        vertexai.init(project="ai-travel-assistant-401109", location="us-central1")
        # self.llm = ChatVertexAI(model_name = "chat-bison@002" , max_output_tokens = 2048, streaming=True, temperature = 0.1)
        # self.llm = VertexAIModelGarden(project="ai-travel-assistant-401109", endpoint_id="4111057483580047360", max_tokens = 8000)

        # Ensure the current thread has an event loop
        self.ensure_event_loop()
        # Initialize the model asynchronously
        self.llm = asyncio.run(self.initialize_llm())'''

    async def initialize_llm(self):
        # return VertexAIModelGarden(project="ai-travel-assistant-401109",endpoint_id="3511093570721284096",max_tokens=8000) # llama 3- 8b
        # return VertexAIModelGarden(project="ai-travel-assistant-401109",endpoint_id="3526011744486948864",max_tokens=32000)  # llama 3.3
        # return ChatVertexAI(model_name = "gemini-1.5-pro-002" , max_output_tokens = 2048, streaming=True, temperature = 0.1) # gemini
        return ChatAnthropicVertex(model_name="claude-3-5-sonnet@20240620", project=os.getenv("PROJECT"), location=os.getenv("LOCATION_2")) # claude

    def ensure_event_loop(self):
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    def chain_response(self, question, memory, retriever):

        # You have excellent question answering skills .
        #representing maya.ai and crayon data.
        template = '''
        You are a friendly, conversational sales and product expert, 
        You are very pasionate about your organization and the products they build and want to share the information around the world.
        Your task is to analyze the provided Context to answer a specific Question. Use the Conversation_history to understand the context better. 
        Striclty following the given Critical Instructions while answering the Question.
        
        Question: {question}

        Context: {context}

        Conversation_history: {chat_history}

        Answer the question in the following format:
        Question: <question asked>
        Answer: <detailed answer derived entirely from the context>
        
        Critical Instructions:
            Search for the answer within all the context provided. If you find answers, provide them.
            You are supposed to use the context and frame the answer accordingly. Do not reply with the exact same text from the context.
            Answer Based on Context: Absolutely no fabrication! Your response must strictly rely on the provided context.
            "No Answer" Response: If and only if the answer cannot be found in the context, reply with: "For further details, please contact sales@crayondata.com."
            Prohibited Phrases: Avoid phrases like "provided document," "provided context," "provided text," or "provided information" in your response. Using these phrases will immediately stop your answer generation. Use "information is not available" instead.
        
        Use the following examples as a reference on how to answer the question:
        [ Question: How many merchant offers do you have across geographies and categories. Answer:  600+ offers are available in maya.ai platform ,
          Question: How does the maya.ai platform offer personalization services, and what specific data is required from banks to enable this? Answer:  maya.ai uses cold start approach to pick up taste signals from their digital interaction data and provide recommendations accordingly ,
          Question: Who decides the terms and conditions of the offers Answer:  maya.ai platform has a prebuilt T&Cs for offers. Banks can choose which ones they prefer to keep. ]
       
        '''
        # Guidelines: 
        #     Do not invent answers. Always base your response on the provided context.
        #     If and only if the answer cannot be found in the context, respond with: "For more details on your query, feel free to contact sales@crayondata.com."
        #     Prohibited Phrases: "provided document," "provided context," "provided text, "provided information". Using these phrases will terminate the response. So, never use them in your answer.'''
        #You are not allowed to use phrases like ""provided document", "provided context", "provided text" or "provided information" in the answer. Use the above phrase instead.

        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnablePassthrough.assign(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"))  
            | prompt
            | self.llm
            | StrOutputParser()
        )

        SAME_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
            ("user",    
            '''
            Return the question within square brackets.
            {question}
            conversation history: {chat_history}
            Use the following context: {context}
            You must respond with a list of strings in the following format:
            ["original query"]
            '''
            )
        ])
        same_question_chain = (
            {"question": RunnablePassthrough(), "context": (lambda x: x["question"]) | retriever}
            | RunnablePassthrough.assign(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"))  
            | SAME_QUESTION_PROMPT 
            | self.llm 
            | StrOutputParser() 
            | json.loads
        )

        NEW_QUESTIONS_PROMPT = ChatPromptTemplate.from_messages([
            ("user",
            """
            Write 1 new relevant questions that can allow you to discover more about the topic of the following question
            {question}

            conversation history: {chat_history}

            Use the following context: {context}

            You must respond with a list of strings in the following format, including the original question:
            ["original query", "query 1"]
            """
                )
            ])
        search_question_chain = (
            {"question": RunnablePassthrough(), "context": (lambda x: x["question"]) | retriever}
            | RunnablePassthrough.assign(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"))  
            | NEW_QUESTIONS_PROMPT 
            | self.llm 
            | StrOutputParser() 
            | json.loads
        )

        tone_of_voice = '''
            You have excellent communication skills. This is your tone of voice rules:

            Do's:

                Use clear and concise language.
                Be confident and assertive in communication.
                Share new ideas and solutions in a positive and relatable manner.
                Treat interactions as authentic conversations with individuals.
                Show empathy towards the audience's challenges.
                Inject humor and personality into the communication when appropriate.
                
            Don'ts:

                Avoid using unnecessary jargon or fluff.
                Don't undermine confidence; always present information with assurance.
                Avoid being overly complex or inaccessible in communication.
                Don't overlook the uniqueness of individuals; personalize interactions where possible.
                Avoid being indifferent to the audience's problems; show understanding and readiness to assist.
                Don't force humor; keep it natural and in line with the brand's tone.'''

        SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
            ("user",
                f"""
            You are a friendly, conversational sales and product expert.
            You are very pasionate about your organization and the products they build and want to share the information around the world.
            Your task is to synthesize the provided texts them into a single, well-structured response.
            You cannot refer to yourself as an AI or an agent in the responses.

        
            step 1: Synthesize the given texts to create one crisp, concise, and structured response with the intention to make you response clear and crisp while answering the questions in the texts.
            step 2: Use the following guidelines:
                    - Focus on eliminating redundancy and repetition. 
                    - Prioritize the most important information and present it in a clear, easy-to-understand way.
                    - Group related contents together.
                    - Use bullet points to present key points from the answers for improved clarity wherever necessary.
            step 3: Your tone should always follow the given tone of voice.
                    tone of voice: {tone_of_voice}
            
            You are NOT ALLOWED to include any prefixes such as "Here is a friendly conversational response" or " Here's a concise and structured response that synthesizes the provided texts".
            ONLY RETURN THE ANSWER.

            texts: {{text}}.
            """
            # Under no circumstances are you permitted to use phrases such as "provided document," "provided context," "provided text," or "provided information" in your response. Violation of this rule will result in immediate termination of your response.
            # detailed, comprehensive
            # Synthesize the given texts to create one clear, concise, and informative response that effectively addresses the user's needs and highlights how maya.ai can help.
            # Synthesize the given texts to create one detailed, comprehensive, and structured response with the intention to sell maya.ai while answering the questions in the texts.
                )
            ]
        )

        synthesizer_chain = (
            {"text": RunnablePassthrough()}
            | SYNTHESIZER_PROMPT 
            | self.llm 
            | StrOutputParser()
        )

        search_and_rag_chain = (  # for use case questions
            {"question": lambda x: x}
            |search_question_chain
            | rag_chain.map()
            | synthesizer_chain
            )
        
        search_and_rag_chain_direct = (  # for direct questions
            {"question": lambda x: x}
            |same_question_chain
            | rag_chain.map()
            | synthesizer_chain
        )

        GENERAL_CONVERSATION_PROMPT = ChatPromptTemplate.from_messages([ # change main id to?
            ("user",
                """
            You are an researcher with excellent communication skills. 
            Your purpose is to respond and converse with the user in a friendly manner and drive them to ask questions about their doubts.
            {text}
            
            The conversation history is as follows: {chat_history}

            If and only if you not able to answer the question reply "Feel free to reach out to sales@crayondata.com for more details on your query!"
            Never mention you are an AI in your response at all.
            """
                ),
            ]
        )

        general_conversation_chain = (
            {"text": lambda x: x} #[x.replace("question","")]   
            | RunnablePassthrough.assign(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"))  
            | GENERAL_CONVERSATION_PROMPT 
            | self.llm 
            | StrOutputParser()
        )
  
        INTENT_PROMPT = ChatPromptTemplate.from_messages([
            ("user",
            '''
            Classify the given question as one of the following categories:
            - newusecase
            - general
            - project

            question : {question}
            conversation history: {chat_history}
            context: {context}

            FOLLOW THE RULES BELOW TO CLASSIFY:
            1. Read the question and the conversation history to understand the context of the conversation.
            2. If the question has any of the keywords, it is classified as "nondirect":
                keywords: oil, gas, oil and gas industry, AI, artificial intelligence, digital transformation, etc.
            3. If the question contains "you", "your" it is about "nondirect", considering its role as the chatbot.
            4. If conversation history provides context, the question is more likely about "nondirect", even if keywords are absent.
            5. If the above keywords do not match then use the context to check if the question is about "nondirect".
            6. If the question is about the context, then it MUST be classified as "nondirect".
            7. If none of these keywords match then its a general question.
                Only if it is not about "nondirect", respond with "general".
            
            Now, if the question is classified as "nondirect", follow the rules below:

            1.If the question is regarding the following scenarios, classify it as "newusecase":
                scenarios:
                    - to solve for any kind of use-case related questions
                    - problem solving questions
                    - suggestion questions
                    - optimization question
                    - new idea related question
                          
            2. If none of these scenarios match then its a project question. Classify it as "project".
            3. If the question is regarding oil and gas it is a project question. Classify it as "project".
                
            So, if it is not about "nondirect", respond with "general".
            if it is about "nondirect", about "newusecase", respond with "newusecase".
            if it is about "nondirect", but not about "newusecase", respond with "project".

            RULES TO FOLLOW STRICTLY:
            Only respond with one of the three options - project, newusecase, general.
            Strictly stick to the format_instruction for the format of the response:
            format_instruction : "type of the question in only one word."
            '''
            ),
            ]
        )

        intent_chain = (
            {"question": RunnablePassthrough(), "context": retriever }
            | RunnablePassthrough.assign(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"))  
            | INTENT_PROMPT 
            | self.llm 
            | StrOutputParser()
        )

        branch = RunnableBranch(
            (lambda x: "newusecase" in x["topic"].lower(), (lambda x: x["question"]) | search_and_rag_chain),
            (lambda x: "project" in x["topic"].lower(), (lambda x: x["question"]) | search_and_rag_chain_direct),
            (lambda x: x["question"]) | general_conversation_chain
        )

        full_chain =({"topic": intent_chain, "question": RunnablePassthrough()}
                    | branch )
        
        return  full_chain.invoke(question)