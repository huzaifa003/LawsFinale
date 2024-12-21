import openai
from ai_utils import query_retreiver
import os
# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def prepare_context(retrieval_results, max_tokens=20000):
    """Prepares context by concatenating retrieved documents up to the token limit."""
    context = ""
    current_length = 0
    for doc, score in retrieval_results:
        doc_content = doc.page_content.strip()
        context += doc_content + "\n\n"
        current_length += len(doc_content.split())
        if current_length >= max_tokens:
            break
    return context

def query_openai(context, query):
    """Queries OpenAI API with the prepared context and query."""
    prompt = f"""
    انتا مساعد ذكي تجيب على الاسئلة باللغة العربية و بشكل واضح بدون أي إضافات.
    context: {context}

    السؤال: {query}
    الإجابة:
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "اأنت مساعد ذكي مصمم كنظام استرجاع معزز بالتوليد (RAG). هدفك هو الإجابة على الأسئلة القانونية المتعلقة بأنظمة وقوانين المملكة العربية السعودية. تعمل بقاعدة معرفية شاملة باللغة العربية، مما يضمن تقديم إجابات دقيقة وذات صلة بالسياق، مع الحفاظ على الطلاقة والوضوح في اللغة العربية."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    print(response)
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    # Simulate retrieval results
    retrieval_results = query_retreiver('ماهي المادة الخامسة عشره من نظام مكافحة الرشاوي؟')
    context = prepare_context(retrieval_results)

    # Query
    query = "ماهي المادة الخامسة عشره من نظام مكافحة الرشاوي؟"
    print("Prepared Context:", context)
    print("Query:", query)

    # Get the result from OpenAI
    result = query_openai(context, query)
    print("Result:", result)
