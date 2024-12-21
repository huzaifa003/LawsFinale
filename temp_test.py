from ai_utils import process_docs, get_compression_retriever, query_retreiver
from langchain_core.runnables import RunnablePassthrough, RunnableMap

def prepare_context(retrieval_results, max_tokens=1500):
    context = ""
    current_length = 0
    for doc, score in retrieval_results:
        doc_content = doc.page_content.strip()
        context += doc_content + "\n\n"
        current_length += len(doc_content.split())
        if current_length >= max_tokens:
            break
    return context

def make_vector_db(input_folder, chunks_folder, vector_db_dir):
    process_docs(input_folder, chunks_folder, vector_db_dir)   
    return

import os
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


if __name__ == "__main__":

    # print(query_retreiver('ماهي المادة الثالثة والعشرون من نظام المتفجرات والمفرقعات؟'))

    # make_vector_db('Folders', 'Chunks', 'VectorDB')
    compression_retriever = get_compression_retriever('VectorDB', 'Chunks')

    context = query_retreiver('ماهي المادة الثالثة والعشرون من نظام المتفجرات والمفرقعات؟')
    context = prepare_context(context)
    context = {"context": context}
    print(context)
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_AI_KEY"))
    

    template = """
    <|system|>
    أنت مساعد ذكي مصمم كنظام استرجاع معزز بالتوليد (RAG). هدفك هو الإجابة على الأسئلة القانونية المتعلقة بأنظمة وقوانين المملكة العربية السعودية. تعمل بقاعدة معرفية شاملة باللغة العربية، مما يضمن تقديم إجابات دقيقة وذات صلة بالسياق، مع الحفاظ على الطلاقة والوضوح في اللغة العربية.
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

    # print(compression_retriever)
    query = "ماهي المادة الثالثة والعشرون من نظام المتفجرات والمفرقعات؟"
    result = qa_chain.invoke(query)
    print(result)



# if __name__ == "__main__":
#     # Retrieve context from your vector database
#     retrieval_results = query_retreiver('ماهي المادة الثالثة والعشرون من نظام المتفجرات والمفرقعات؟')
#     context = prepare_context(retrieval_results)
#     print(context)

#     # Define the LLM
#     llm = ChatOpenAI(temperature=0.7, model_name="gpt-4", api_key="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")

#     # Define the prompt template
#     template = """
#     <|system|>
#     انتا مساعد ذكي تجيب على الاسئلة باللغة العربية و بشكل واضح بدون أي إضافات
#     context: {context}
#     </s>
#     <|user|>
#     {query}
#     </s>
#     <|assistant|>
#     """ 
#     prompt = ChatPromptTemplate.from_template(template)
#     output_parser = StrOutputParser()

#     # Correct the RunnableMap to ensure proper input formatting
#     context_runnable = RunnableMap(
#         {"context": lambda _: context, "query": RunnablePassthrough()}
#     )

#     # Combine into the full pipeline
#     qa_chain = context_runnable | prompt | llm | output_parser

#     query = "ماهي المادة الثالثة والعشرون من نظام المتفجرات والمفرقعات؟"
#     print("Prepared Context:", context)
#     print("Query:", query)
#     print("Prompt:", prompt.format(context=context, query=query))


#     # Run the chain with a query
    
#     result = qa_chain.invoke({"context": context, "query": query})
#     print(result)
