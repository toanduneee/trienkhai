import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

TARGET_MODELS = {
    "dolphin":  "ollama/dolphin-llama3",              # local — ít align, dễ bypass
    "groq":    "groq/llama-3.3-70b-versatile", # Groq cloud — free tier
    "gemini":  "gemini/gemini-2.0-flash",     # Google AI Studio — free tier
}


def call_target(model_key: str, prompt: str, system_prompt: str = "") -> str:
    """
    Gọi target model và trả về response string.

    Args:
        model_key:     "dolphin" | "groq" | "gemini"
        prompt:        Prompt tấn công.
        system_prompt: System prompt của target app (nếu có).

    Returns:
        Response string từ model.
    """
    model_name = TARGET_MODELS.get(model_key)
    if not model_name:
        raise ValueError(
            f"Unknown model key: '{model_key}'. "
            f"Available: {list(TARGET_MODELS.keys())}"
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"model": model_name, "messages": messages}

    # Ollama cần chỉ rõ api_base
    if model_key == "dolphin":
        kwargs["api_base"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    response = completion(**kwargs)
    return response.choices[0].message.content


def make_caller(model_key: str, system_prompt: str = ""):
    """
    Tạo callable dùng trong vòng lặp PAIR.

    Returns:
        f(prompt: str) -> str
    """
    def caller(prompt: str) -> str:
        return call_target(model_key, prompt, system_prompt)
    return caller