import google.generativeai as genai

from resources.config import MAX_CONTEXT_LENGTH, MODEL_NAME


def check_html_token_limit(gemini_api_key, html_documents):
    joined_html_documents = ""
    for idx, html in enumerate(html_documents):
        joined_html_documents += f"\n{idx + 1}. HTML document\n" + html

    try:
        total_tokens = get_llm_token_count(gemini_api_key, joined_html_documents)
        if total_tokens and (total_tokens > MAX_CONTEXT_LENGTH):
            return False
    except Exception as e:
        raise Exception("Could not calculate total HTML tokens with gemini API.", e)

    return total_tokens


def get_llm_token_count(gemini_api_key, text: str):
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(MODEL_NAME)

        token_length_str = str(model.count_tokens(text))
        return int(token_length_str.split(":")[1].strip())
    except Exception as e:
        raise Exception(e)
