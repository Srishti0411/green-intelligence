import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# System Prompt
SYSTEM_PROMPT = """
You are an AI sustainability expert working with machine learning systems.

Review the ML workload details below and provide clear, practical sustainability
insights.

Context:
- Optimization Goal: {goal}
- Hardware: {hardware}
- Model: {model}
- Avg Energy (kWh): {avg_energy}
- Avg CO2 (kg): {avg_co2}
- Most Efficient Device: {efficient_device}

Guidelines:
- Give exactly 3 insights
- Each insight should include a technical observation and a concrete recommendation
- Keep it concise and technical
- Use bullet points
- No emojis or generic advice
"""

# Public API
def generate_ai_insights(context: dict) -> str:
    """
    Generate sustainability insights for an ML workload.
    """

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ AI insights unavailable (GROQ_API_KEY not set)."

    client = Groq(api_key=api_key)

    required_keys = [
        "goal", "hardware", "model",
        "avg_energy", "avg_co2", "efficient_device"
    ]
    missing = [k for k in required_keys if k not in context]
    if missing:
        return f"⚠️ Missing context fields: {', '.join(missing)}"

    prompt = SYSTEM_PROMPT.format(
        goal=context["goal"],
        hardware=context["hardware"],
        model=context["model"],
        avg_energy=context["avg_energy"],
        avg_co2=context["avg_co2"],
        efficient_device=context["efficient_device"],
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    except Exception:
        return "⚠️ AI insights could not be generated at this time."
