import os
from flask import Flask, request, jsonify, render_template
from ai_utils import process_docs, get_compression_retriever
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from flask_cors import CORS

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


# Global variables
compression_retriever = None
llm = None
qa_chain = None


"""Initialize the vector database and retriever."""



vector_db_dir = 'VectorDB'
chunks_folder = 'Chunks'

try:
    # Process the documents and create the vector database
    # process_docs(input_folder, chunks_folder, vector_db_dir)
    compression_retriever = get_compression_retriever(vector_db_dir, chunks_folder)
    print(os.getenv("OPENAI_AI_KEY"))
    # Initialize the LLM
    llm = ChatOpenAI(temperature=0.7, max_tokens=256, model_name="gpt-4o-mini", 
                        api_key=str(os.getenv("OPENAI_AI_KEY")))

    # Define the QA chain
    template = """
    <|system|>
    انتا مساعد ذكي تجيب على الاسئلة باللغة العربية و بشكل واضح بدون أي إضافات
    context: {context}
    </s>
    <|user|>
    {query}
    </s>
    <|assistant|>
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    qa_chain = (
        {"context": compression_retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

    
except Exception as e:
    print(f"Error during initialization: {e}")


@app.route('/')
def home():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    """Handle user queries and return answers."""
    global qa_chain

    if not qa_chain:
        return jsonify({"error": "System is not initialized. Call /init first."}), 400

    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        result = qa_chain.invoke(query)
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
