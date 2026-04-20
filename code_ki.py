mport torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import io
import contextlib

model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2" # 2x schneller wenn installiert
)

SYSTEM_PROMPT = """Du bist eine krasse Coding-KI. Du schreibst sauberen, kommentierten, produktionsreifen Code.
Wenn der User nach Code fragt, antworte NUR mit Code + kurzer Erklärung.
Nutze immer Markdown Code-Blöcke. Wenn du!run siehst, führst du mentalen Code aus und gibst Output an."""

chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

def run_python(code: str) -> str:
    """Lässt die KI Code sicher ausführen und fängt Output ab"""
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {})
        return output.getvalue() or "Code lief ohne Output."
    except Exception as e:
        return f"Fehler: {e}"

def code_ki_stream(user_input: str):
    global chat_history

    # Spezialkommando:!run führt letzten Code-Block aus
    if user_input.strip() == "!run":
        last_code = ""
        for msg in reversed(chat_history):
            if "```python" in msg["content"]:
                last_code = msg["content"].split("```python")[1].split("```")[0]
                break
        if last_code:
            result = run_python(last_code)
            print(f"\n[Output]\n{result}")
            chat_history.append({"role": "user", "content": "!run"})
            chat_history.append({"role": "assistant", "content": f"[Output]\n{result}"})
        else:
            print("Kein Python-Code zum Ausführen gefunden.")
        return

    chat_history.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(
        chat_history,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        inputs=inputs,
        streamer=streamer,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.1
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Assistent: ", end="", flush=True)
    full_response = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        full_response += new_text

    chat_history.append({"role": "assistant", "content": full_response})
    print() # newline

# Hauptloop
if __name__ == "__main__":
    print("Krasse Code-KI gestartet. Tippe 'exit' zum Beenden. Mit '!run' führst du den letzten Code aus.\n")
    while True:
        user_in = input("Du: ")
        if user_in.lower() == "exit":
            break
        code_ki_stream(user_in)
