from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

prompt_template = """You are an Apple Support assistant trained to help users solve iPhone issues.
Try to answer using the provided context below. If the answer is not in the context, then try to answer yourself, if you don't know 
say "I am not sure about that, please check the official Apple Support website for more information."

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

# 1. Load FAISS vector DB
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

# 2. Load LLM from Ollama (e.g. llama3)
llm = ChatOllama(model="llama3.2", temperature=0)

# 3. Add memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 4. Build RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    memory=memory,
    verbose=True
)

# 5. Chat loop
print("Hi! I am a virtual iPhone support representative. Ask me something about iPhone support articles. Type 'exit' to quit.\n")

while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        break

    response = qa_chain.run(question)
    print(f"Bot: {response}\n")
