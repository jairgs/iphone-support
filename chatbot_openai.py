from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings

# 1. Custom prompt
prompt_template = """You are an Apple Support assistant trained to help users solve iPhone issues.
Try to answer using the provided context below. If the answer is not in the context, then try to answer yourself; 
if you don't know, say "I am not sure about that, please check the official Apple Support website for more information."

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
"""

custom_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=prompt_template,
)

# 2. Load FAISS vector DB (built with OpenAIEmbeddings)
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

# 3. Instantiate OpenAI chat model
#    Make sure you’ve set OPENAI_API_KEY in your environment!
llm = ChatOpenAI(
    model_name="gpt-4",  # or "gpt-4" if you have access
    temperature=0
)

# 4. Add conversational memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 5. Build the RAG chain (passing our custom prompt into the combine-docs step)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    verbose=True
)

# 6. Simple CLI chat loop
print("🛠️  Apple iPhone Support Assistant (OpenAI + FAISS)\nType ‘exit’ to quit.\n")
while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        break
    answer = qa_chain.run(question)
    print(f"\nBot: {answer}\n")
