import gradio as gr
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings #OllamaEmbeddings
from langchain.chat_models import ChatOpenAI # ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load vector store
db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

def retrieve_context(query, k=3):
    docs = db.similarity_search(query, k=k)
    return "\n\n---\n\n".join([doc.page_content for doc in docs]), docs

# Helper to create LLM
def initialize_llm(model_choice):
    # if model_choice == "llama3.2":
    #     return ChatOllama(model="llama3.2", temperature=0.2)
    if model_choice == "gpt-3.5-turbo":
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    elif model_choice == "gpt-4":
        return ChatOpenAI(model_name="gpt-4", temperature=0)
    else:
        raise ValueError("Unsupported model.")

# Main chat streaming generator
def stream_chat(user_message, history, model_choice, state):
    # state = (llm_instance, prev_model)
    llm, prev_model = state or (None, None)

    # Reinitialize if model changed
    if llm is None or model_choice != prev_model:
        llm = initialize_llm(model_choice)
        prev_model = model_choice

    # Get RAG context
    context, _ = retrieve_context(user_message)

    # LangChain prompt structure
    system_prompt = (
        "You are a helpful assistant for Apple iPhone users. "
        "Use the support documentation below to answer user questions as accurately as possible.\n\n"
        f"{context}"
    )

    messages = [SystemMessage(content=system_prompt)]
    for user, bot in history:
        messages.append(HumanMessage(content=user))
        messages.append(AIMessage(content=bot))
    messages.append(HumanMessage(content=user_message))

    stream = llm.stream(messages)
    reply = ""
    for chunk in stream:
        reply += chunk.content
        yield history + [(user_message, reply)], (llm, prev_model)

# Gradio app
with gr.Blocks() as demo:
    gr.Markdown(
    "<h1 style='text-align: center;'>📱 Apple iPhone Support Assistant (using RAG)</h1>"
)

    model_choice = gr.Dropdown(["gpt-3.5-turbo", "gpt-4"], label="Select Model", value="gpt-4")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question", placeholder="e.g. How do I reset my iPhone?")
    llm_state = gr.State(value=None)

    def respond(user_message, history, model_choice, llm_state):
        for updated_history, new_state in stream_chat(user_message, history, model_choice, llm_state):
            yield updated_history, new_state, ""

    msg.submit(
        respond,
        [msg, chatbot, model_choice, llm_state],
        [chatbot, llm_state, msg]
    )

    gr.Markdown("This assistant uses RAG with iPhone support docs. Select your model and start chatting!")

demo.launch(server_name="0.0.0.0")
