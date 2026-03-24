import base64
import json
import hashlib
import io

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

from agent import ChefAgent
from config import APP_PASSWORD, APP_USERNAME, OPENAI_API_KEY


SUGGESTED_PROMPTS = [
    "How do I make masala dosa at home?",
    "Suggest a quick North Indian vegetarian dinner.",
    "What can I cook with paneer and tomato?",
    "Give me a beginner-friendly South Indian recipe.",
]
RECIPE_OF_THE_DAY_QUESTION = "Recipe of the Day"
RECIPE_OF_THE_DAY_ANSWER = """
Today's recipe is Vegetable Pulao.

Ingredients:
- 1 cup basmati rice
- 2 cups water
- 1 sliced onion
- 1 chopped carrot
- 1/2 cup peas
- 1 sliced beans handful
- 1 tomato chopped
- 1 tsp ginger-garlic paste
- 2 tbsp oil or ghee
- 1 bay leaf
- 1 small cinnamon piece
- 2 cloves
- 2 cardamom pods
- Salt to taste
- 1/2 tsp garam masala
- 2 tbsp chopped coriander

Steps:
1. Wash and soak the basmati rice for 15 minutes.
2. Heat oil or ghee in a pan and add bay leaf, cinnamon, cloves, and cardamom.
3. Add onion and saute until lightly golden.
4. Stir in ginger-garlic paste, then add tomato and cook until soft.
5. Add carrot, peas, beans, salt, and garam masala. Cook for 2 to 3 minutes.
6. Add soaked rice and gently mix.
7. Pour in water, cover, and cook until the rice is soft and fluffy.
8. Rest for 5 minutes, then fluff and finish with coriander.

Serve hot with raita, pickle, or plain yogurt.
""".strip()
PROMPT_PLACEHOLDER = "Choose a quick prompt"
VOICE_TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
VOICE_TTS_MODEL = "gpt-4o-mini-tts"
VOICE_TTS_VOICE = "sage"
RECIPE_IMAGE_MODEL = "gpt-image-1"
VOICE_LANGUAGES = {
    "English": {"transcribe": "en", "reply_name": "English"},
    "Hindi": {"transcribe": "hi", "reply_name": "Hindi"},
    "Malayalam": {"transcribe": "ml", "reply_name": "Malayalam"},
    "Marathi": {"transcribe": "mr", "reply_name": "Marathi"},
    "Tamil": {"transcribe": "ta", "reply_name": "Tamil"},
    "Telugu": {"transcribe": "te", "reply_name": "Telugu"},
    "Kannada": {"transcribe": "kn", "reply_name": "Kannada"},
}
VOICE_TTS_INSTRUCTIONS = (
    "Speak like a warm, expressive, encouraging chef. "
    "Sound lively, friendly, and a little theatrical when describing recipes, "
    "while staying clear and natural."
)
VOICE_QUERY_GUIDANCE_MESSAGE = "Question nor understood, ask about any Indian vegeterian recipies"


def normalize_query_param(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def attempt_login():
    if not APP_USERNAME or not APP_PASSWORD:
        st.session_state["is_authenticated"] = False
        st.session_state["login_error"] = "Login is not configured. Set APP_USERNAME and APP_PASSWORD first."
        return

    entered_username = st.session_state.get("login_username", "").strip()
    entered_password = st.session_state.get("login_password", "")
    if entered_username == APP_USERNAME and entered_password == APP_PASSWORD:
        st.session_state["is_authenticated"] = True
        st.session_state["login_error"] = ""
        st.session_state["login_password"] = ""
    else:
        st.session_state["is_authenticated"] = False
        st.session_state["login_error"] = "Invalid user ID or password."


def apply_selected_prompt():
    prompt = st.session_state.get("selected_prompt", PROMPT_PLACEHOLDER)
    if prompt != PROMPT_PLACEHOLDER:
        queue_text_query(prompt, add_to_chat=False)


def get_latest_state_prefix(source):
    return "voice" if source == "voice" else "text"


def clear_latest_response(source=None):
    prefixes = [get_latest_state_prefix(source)] if source else ["text", "voice"]
    for prefix in prefixes:
        st.session_state[f"{prefix}_latest_question"] = ""
        st.session_state[f"{prefix}_latest_answer"] = ""
        st.session_state[f"{prefix}_latest_recipe_image_key"] = ""
        st.session_state[f"{prefix}_latest_recipe_image_b64"] = ""
        st.session_state[f"{prefix}_latest_recipe_image_error"] = ""


def show_recipe_of_the_day():
    clear_latest_response(source="chat")
    st.session_state["text_latest_question"] = RECIPE_OF_THE_DAY_QUESTION
    st.session_state["text_latest_answer"] = RECIPE_OF_THE_DAY_ANSWER


def queue_text_query(query, add_to_chat):
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return False

    clear_latest_response(source="chat")
    st.session_state["pending_text_query"] = cleaned_query
    st.session_state["pending_text_add_to_chat"] = add_to_chat
    return True


def clear_chat():
    st.session_state["chat_history"] = []
    st.session_state["voice_history"] = []
    st.session_state["pending_prompt"] = ""
    st.session_state["pending_text_query"] = ""
    st.session_state["pending_text_add_to_chat"] = False
    st.session_state["voice_input_value"] = ""
    st.session_state["voice_question_selection"] = ""
    st.session_state["text_question_selection"] = ""
    st.session_state["voice_widget_reset"] = st.session_state.get("voice_widget_reset", 0) + 1
    st.session_state["ask_chef_input"] = ""
    st.session_state["selected_prompt"] = PROMPT_PLACEHOLDER
    clear_latest_response()
    st.session_state["voice_last_question"] = ""
    st.session_state["voice_last_answer"] = ""
    st.session_state["voice_last_error"] = ""
    st.session_state["voice_status_message"] = "Ready to record a recipe question."
    st.session_state["last_voice_audio_hash"] = ""
    st.session_state["auto_speak_pending"] = False
    st.session_state["speak_text_once"] = ""
    st.session_state["last_spoken_text"] = ""
    st.session_state["last_spoken_audio_b64"] = ""
    st.session_state["last_spoken_signature"] = ""
    st.session_state["pending_stop_voice"] = True


def get_voice_language_settings():
    selected_language = st.session_state.get("voice_language", "English")
    return VOICE_LANGUAGES.get(selected_language, VOICE_LANGUAGES["English"])


def get_voice_transcription_prompt():
    selected_language = st.session_state.get("voice_language", "English")
    return (
        f"The speaker is speaking {selected_language}. "
        f"Transcribe strictly in {selected_language}. "
        f"Do not switch to another Indian language or script. "
        f"Do not transliterate into a different script. "
        f"If the speech mentions Indian vegetarian dishes, preserve those words naturally in {selected_language}."
    )


def get_conversation_context(history_key, max_turns=4):
    history = st.session_state.get(history_key, []) or []
    if not history:
        return ""

    turns = []
    for exchange in history[-max_turns:]:
        question = (exchange.get("question") or "").strip()
        answer = (exchange.get("answer") or "").strip()
        if question:
            turns.append(f"User: {question}")
        if answer:
            turns.append(f"AI Chef: {answer}")
    return "\n".join(turns)


def get_text_question_options():
    options = []
    seen = set()

    for exchange in reversed(st.session_state.get("chat_history", [])):
        if exchange.get("source") != "chat":
            continue
        question = (exchange.get("question") or "").strip()
        if question and question not in seen:
            options.append(question)
            seen.add(question)
    return options


def build_agent_query(query, source="chat", history_key="chat_history"):
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return ""

    conversation_context = get_conversation_context(history_key)
    context_prefix = ""
    if conversation_context:
        context_prefix = (
            "Continue the same recipe conversation using the recent chat context below. "
            "If the user asks for a change like sweeter, spicier, thicker, or simpler, "
            "apply that change to the current recipe instead of starting a new one.\n\n"
            f"Recent conversation:\n{conversation_context}\n\n"
        )

    if source != "voice":
        return f"{context_prefix}User question: {cleaned_query}".strip()

    language_settings = get_voice_language_settings()
    reply_language = language_settings["reply_name"]
    return (
        f"{context_prefix}"
        f"Answer in {reply_language}. "
        "Give the same level of detail and clarity as a strong English recipe answer. "
        "Include ingredients, quantities when possible, practical cooking steps, and helpful tips or substitutions when relevant. "
        "Do not make the answer shorter just because it is in another language. "
        "Keep ingredient names and steps natural for that language. "
        f"User question: {cleaned_query}"
    )


def run_agent_query(query, add_to_chat=True, history_key="chat_history", source="chat"):
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return None, "Enter a cooking question to get a recommendation."

    agent_query = build_agent_query(cleaned_query, source=source, history_key=history_key)
    answer = st.session_state.agent.ask(agent_query) or "No answer was generated."
    latest_prefix = get_latest_state_prefix(source)
    st.session_state[f"{latest_prefix}_latest_question"] = cleaned_query
    st.session_state[f"{latest_prefix}_latest_answer"] = answer
    if source == "voice":
        st.session_state["voice_last_question"] = cleaned_query
        st.session_state["voice_last_answer"] = answer
        st.session_state["voice_last_error"] = ""
    if add_to_chat:
        st.session_state[history_key].append(
            {
                "source": source,
                "question": cleaned_query,
                "answer": answer,
            }
        )
    return answer, None


def process_pending_prompt(add_to_chat=True, history_key="chat_history", source="chat"):
    pending_query = st.session_state.get("pending_prompt", "").strip()
    if not pending_query:
        return None

    try:
        _, error_message = run_agent_query(
            pending_query,
            add_to_chat=add_to_chat,
            history_key=history_key,
            source=source,
        )
        st.session_state["pending_prompt"] = ""
        if not error_message and st.session_state.get("auto_speak_pending") and st.session_state.get("voice_latest_answer"):
            st.session_state["speak_text_once"] = st.session_state["voice_latest_answer"]
        st.session_state["auto_speak_pending"] = False
        return error_message
    except Exception as exc:
        st.session_state["pending_prompt"] = ""
        st.session_state["auto_speak_pending"] = False
        if source == "voice":
            st.session_state["voice_last_question"] = pending_query
            st.session_state["voice_last_answer"] = ""
            st.session_state["voice_last_error"] = f"Could not generate a response: {exc}"
        return f"Could not generate a response: {exc}"


def speak_text(text):
    clipped_text = (text or "").strip()
    if not clipped_text:
        return

    selected_language = st.session_state.get("voice_language", "English")
    cached_signature = st.session_state.get("last_spoken_signature", "")
    cached_audio_b64 = st.session_state.get("last_spoken_audio_b64", "")
    current_signature = f"{selected_language}:{clipped_text}"
    if current_signature == cached_signature and cached_audio_b64:
        audio_b64 = cached_audio_b64
    else:
        response = st.session_state.openai_client.audio.speech.create(
            input=clipped_text[:2000],
            model=VOICE_TTS_MODEL,
            voice=VOICE_TTS_VOICE,
            instructions=f"{VOICE_TTS_INSTRUCTIONS} Speak the response in {selected_language}.",
            response_format="mp3",
            speed=1.0,
        )
        audio_bytes = response.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        st.session_state["last_spoken_text"] = clipped_text
        st.session_state["last_spoken_audio_b64"] = audio_b64
        st.session_state["last_spoken_signature"] = current_signature

    payload = json.dumps(audio_b64)
    components.html(
        f"""
<script>
const audioBase64 = {payload};
const avatar = window.parent.document.getElementById('chef-avatar-voice');
if (window.parent.aiChefAudio) {{
  try {{
    window.parent.aiChefAudio.pause();
    window.parent.aiChefAudio.currentTime = 0;
  }} catch (e) {{}}
}}
if (audioBase64) {{
  const audio = new Audio(`data:audio/mp3;base64,${{audioBase64}}`);
  window.parent.aiChefAudio = audio;
  audio.onplay = () => avatar && avatar.classList.add('speaking');
  audio.onended = () => avatar && avatar.classList.remove('speaking');
  audio.onpause = () => avatar && avatar.classList.remove('speaking');
  audio.play().catch(() => {{
    if (avatar) avatar.classList.remove('speaking');
  }});
}}
</script>
""",
        height=0,
    )


def stop_speaking():
    components.html(
        """
<script>
const avatar = window.parent.document.getElementById('chef-avatar-voice');
if (window.parent.aiChefAudio) {
  try {
    window.parent.aiChefAudio.pause();
    window.parent.aiChefAudio.currentTime = 0;
  } catch (e) {}
}
if (avatar) {
  avatar.classList.remove('speaking');
}
</script>
""",
        height=0,
    )


def process_pending_voice_stop():
    if not st.session_state.get("pending_stop_voice"):
        return

    stop_speaking()
    st.session_state["pending_stop_voice"] = False


def transcribe_voice_clip(audio_clip):
    if audio_clip is None:
        return "", "Record a voice question first."

    audio_bytes = audio_clip.getvalue()
    if not audio_bytes:
        return "", "The recorded clip was empty. Please try again."

    audio_buffer = io.BytesIO(audio_bytes)
    audio_buffer.name = getattr(audio_clip, "name", None) or "voice_question.wav"
    language_settings = get_voice_language_settings()
    transcript = st.session_state.openai_client.audio.transcriptions.create(
        file=audio_buffer,
        model=VOICE_TRANSCRIBE_MODEL,
        response_format="text",
        language=language_settings["transcribe"],
        prompt=get_voice_transcription_prompt(),
        temperature=0,
    )
    return str(transcript).strip(), None


def get_latest_exchange(preferred_history_key=None):
    latest_prefix = "voice" if preferred_history_key == "voice_history" else "text"
    latest_question = st.session_state.get(f"{latest_prefix}_latest_question", "")
    latest_answer = st.session_state.get(f"{latest_prefix}_latest_answer", "")
    if latest_question or latest_answer:
        return latest_question, latest_answer

    history_key = preferred_history_key or "chat_history"
    history = st.session_state.get(history_key) or []
    if history:
        last_exchange = history[-1]
        return last_exchange.get("question", ""), last_exchange.get("answer", "")
    return "", ""


def get_voice_question_options():
    options = []
    seen = set()

    latest_value = st.session_state.get("voice_input_value", "").strip()
    if latest_value and latest_value not in seen:
        options.append(latest_value)
        seen.add(latest_value)

    for exchange in reversed(st.session_state.get("voice_history", [])):
        question = (exchange.get("question") or "").strip()
        if question and question not in seen:
            options.append(question)
            seen.add(question)
    return options


def ensure_recipe_visual(question, answer, source="chat"):
    latest_prefix = get_latest_state_prefix(source)
    prompt_key = hashlib.sha1(f"{question}\n{answer}".encode("utf-8")).hexdigest()
    if st.session_state.get(f"{latest_prefix}_latest_recipe_image_key") == prompt_key:
        return

    prompt = (
        "Create a realistic, appetizing food photo of the recipe described below. "
        "Show a plated Indian vegetarian dish with natural lighting and relevant garnishes. "
        "Avoid text overlays, labels, watermarks, split panels, or collage layouts.\n\n"
        f"Recipe question: {question}\n"
        f"Recipe answer: {answer[:1200]}"
    )

    response = st.session_state.openai_client.images.generate(
        model=RECIPE_IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        quality="low",
        output_format="png",
    )
    image_b64 = response.data[0].b64_json
    st.session_state[f"{latest_prefix}_latest_recipe_image_key"] = prompt_key
    st.session_state[f"{latest_prefix}_latest_recipe_image_b64"] = image_b64
    st.session_state[f"{latest_prefix}_latest_recipe_image_error"] = ""


def render_recipe_visuals(question, answer, source="chat", show_heading=True):
    latest_prefix = get_latest_state_prefix(source)
    if show_heading:
        st.markdown('<div class="section-heading">Recipe Visuals</div>', unsafe_allow_html=True)
        st.caption("A generated visual for the current recipe. This uses OpenAI image generation in a low-cost mode.")

    try:
        ensure_recipe_visual(question, answer, source=source)
    except Exception as exc:
        st.session_state[f"{latest_prefix}_latest_recipe_image_error"] = f"Recipe image could not be generated: {exc}"

    image_error = st.session_state.get(f"{latest_prefix}_latest_recipe_image_error", "")
    image_b64 = st.session_state.get(f"{latest_prefix}_latest_recipe_image_b64", "")
    if image_error:
        st.warning(image_error)
        return
    if image_b64:
        st.image(base64.b64decode(image_b64), use_container_width=True)


st.set_page_config(
    page_title="AI Chef",
    page_icon="Chef",
    layout="wide",
)

st.markdown(
    """
<style>
.stApp {
    background:
        linear-gradient(180deg, #f8f3ea 0%, #fcfaf5 52%, #f2ede3 100%);
}

.block-container {
    padding-top: 4.8rem;
    padding-bottom: 2rem;
    max-width: 96vw;
}

.hero-shell {
    background: linear-gradient(135deg, #16322f 0%, #28554f 52%, #416f62 100%);
    border-radius: 28px;
    padding: 1rem 1.2rem;
    color: #fff8ee;
    box-shadow: 0 24px 50px rgba(22, 50, 47, 0.24);
    margin-bottom: 1.2rem;
    text-align: center;
}

.hero-main {
    width: 100%;
}

.hero-title {
    font-size: 2rem;
    line-height: 1.1;
    font-weight: 800;
    margin: 0;
}

.hero-avatar {
    width: 5rem;
    height: 5rem;
    border-radius: 50%;
    background: linear-gradient(180deg, #d9732f 0%, #8f3e17 100%);
    border: 3px solid rgba(96, 41, 12, 0.28);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 14px 28px rgba(73, 31, 12, 0.24);
    position: relative;
    overflow: hidden;
}

.hero-avatar.listening {
    animation: chefPulse 1s ease-in-out infinite;
    box-shadow: 0 0 0 6px rgba(223, 155, 80, 0.22), 0 14px 28px rgba(15, 30, 28, 0.22);
}

.hero-avatar.speaking {
    animation: chefPulse 0.8s ease-in-out infinite;
    box-shadow: 0 0 0 8px rgba(201, 112, 30, 0.18), 0 14px 28px rgba(15, 30, 28, 0.22);
}

@keyframes chefPulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.chef-hat {
    position: absolute;
    top: 0.45rem;
    left: 1.1rem;
    width: 2.8rem;
    height: 1.15rem;
    background: #fffdf8;
    border-radius: 1rem 1rem 0.6rem 0.6rem;
}

.chef-hat::before,
.chef-hat::after {
    content: "";
    position: absolute;
    top: -0.45rem;
    width: 1rem;
    height: 1rem;
    background: #fffdf8;
    border-radius: 50%;
}

.chef-hat::before {
    left: 0.15rem;
}

.chef-hat::after {
    right: 0.15rem;
}

.chef-face {
    position: absolute;
    top: 1.45rem;
    left: 1.2rem;
    width: 2.6rem;
    height: 2.6rem;
    background: #f5d2b2;
    border-radius: 50%;
}

.chef-eye {
    position: absolute;
    top: 0.95rem;
    width: 0.22rem;
    height: 0.22rem;
    background: #2a170f;
    border-radius: 50%;
}

.chef-eye.left {
    left: 0.72rem;
}

.chef-eye.right {
    right: 0.72rem;
}

.chef-smile {
    position: absolute;
    left: 0.88rem;
    top: 1.35rem;
    width: 0.82rem;
    height: 0.42rem;
    border-bottom: 2px solid #6d2f17;
    border-radius: 0 0 0.8rem 0.8rem;
}

.hero-avatar.speaking .chef-smile {
    animation: chefTalk 0.22s ease-in-out infinite alternate;
}

@keyframes chefTalk {
    0% {
        height: 0.18rem;
        border-bottom-width: 2px;
        transform: translateY(0);
    }
    100% {
        height: 0.62rem;
        border-bottom-width: 3px;
        transform: translateY(0.04rem);
    }
}

.avatar-tip {
    margin-top: 0.35rem;
    font-size: 0.74rem;
    color: rgba(255, 248, 238, 0.84);
}

.hero-speech {
    max-width: 9rem;
    background: rgba(255, 248, 238, 0.12);
    border: 1px solid rgba(255, 248, 238, 0.2);
    border-radius: 16px;
    padding: 0.55rem 0.7rem;
    font-size: 0.8rem;
    line-height: 1.35;
    color: rgba(255, 248, 238, 0.92);
}

.agent-avatar-card {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid #e8dcc7;
    border-radius: 20px;
    padding: 0.95rem;
    box-shadow: 0 12px 26px rgba(89, 63, 26, 0.08);
    margin-bottom: 0.9rem;
}

.agent-avatar-shell {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.7rem;
}

.agent-avatar-copy {
    color: #6f624f;
    font-size: 0.88rem;
    line-height: 1.4;
}

.voice-avatar-card {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 4.4rem;
    padding: 0;
    margin-bottom: 0.2rem;
    background: transparent;
    border: 0;
    box-shadow: none;
}

.voice-avatar-card .hero-avatar {
    width: 3.35rem;
    height: 3.35rem;
    border-width: 2px;
}

.voice-avatar-card .chef-hat {
    top: 0.28rem;
    left: 0.74rem;
    width: 1.92rem;
    height: 0.78rem;
}

.voice-avatar-card .chef-face {
    top: 0.95rem;
    left: 0.8rem;
    width: 1.72rem;
    height: 1.72rem;
}

.voice-avatar-card .chef-eye {
    top: 0.62rem;
    width: 0.16rem;
    height: 0.16rem;
}

.voice-avatar-card .chef-eye.left {
    left: 0.48rem;
}

.voice-avatar-card .chef-eye.right {
    right: 0.48rem;
}

.voice-avatar-card .chef-smile {
    left: 0.58rem;
    top: 0.9rem;
    width: 0.56rem;
    height: 0.26rem;
    border-bottom-width: 1.5px;
}

.voice-language-label {
    font-size: 0.82rem;
    font-weight: 700;
    color: #7d613e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.35rem;
}

.top-control-label {
    font-size: 0.82rem;
    font-weight: 700;
    color: #7d613e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.35rem;
    min-height: 1rem;
}

.top-control-label.centered {
    text-align: center;
}

.voice-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.38rem 0.62rem;
    border-radius: 999px;
    background: #fff6e7;
    border: 1px solid #ead2a9;
    color: #6a4620;
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.voice-dot {
    width: 0.55rem;
    height: 0.55rem;
    border-radius: 50%;
    background: #d08a3e;
}

.voice-status-pill.listening .voice-dot {
    background: #cf5a1e;
    box-shadow: 0 0 0 6px rgba(207, 90, 30, 0.16);
}


.side-card,
.chat-card,
.answer-card {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid #e8dcc7;
    border-radius: 20px;
    box-shadow: 0 12px 26px rgba(89, 63, 26, 0.08);
}

.side-card {
    padding: 1rem;
}

.chat-card,
.answer-card {
    padding: 1rem 1rem 0.85rem 1rem;
    min-height: 24rem;
}

.latest-answer-shell {
    background: #fffaf2;
    border: 1px solid #ead8bf;
    border-radius: 16px;
    padding: 0.9rem 1rem;
    color: #2f2418;
}

.latest-answer-shell p,
.latest-answer-shell li,
.latest-answer-shell span,
.latest-answer-shell div {
    color: #2f2418 !important;
}

.section-heading {
    color: #5e3514 !important;
    font-size: 1.35rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.card-title {
    font-size: 1rem;
    font-weight: 800;
    color: #5e3514;
    margin-bottom: 0.25rem;
}

.card-copy {
    color: #6f624f;
    font-size: 0.93rem;
    line-height: 1.45;
    margin-bottom: 0.75rem;
    font-weight: 700;
}

.prompt-label {
    font-size: 0.82rem;
    font-weight: 700;
    color: #7d613e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.35rem;
}

.latest-question-text {
    color: #b85a16;
    font-size: 0.9rem;
    font-weight: 800;
    margin-bottom: 0.45rem;
}

.voice-help-text {
    color: #6a4620 !important;
    font-size: 0.82rem;
    font-weight: 700;
    line-height: 1.35;
    margin: 0.15rem 0 0.4rem 0;
}

div[data-testid="stTextInput"] input {
    background: #fffaf2;
    color: #2f2418;
    -webkit-text-fill-color: #2f2418;
    caret-color: #2f2418;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #89715b;
    -webkit-text-fill-color: #89715b;
}

div[data-testid="stTextInput"] > div {
    background: #fffaf2;
}

div[data-testid="stButton"] > button {
    border-radius: 14px;
    border: 1px solid #cb8740;
    background: linear-gradient(180deg, #df9b50 0%, #c9701e 100%);
    color: #ffffff;
    font-weight: 800;
}

div[data-testid="stButton"] > button:hover {
    background: linear-gradient(180deg, #d38d40 0%, #b75f11 100%);
    color: #ffffff;
}

[data-testid="stChatMessage"] {
    background: rgba(255, 250, 242, 0.96);
    border: 1px solid #ead8bf;
    border-radius: 18px;
    padding: 0.25rem 0.35rem;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #2f2418 !important;
}

[data-testid="stMarkdownContainer"] code {
    background: linear-gradient(180deg, #fff3df 0%, #fde6c5 100%);
    color: #9a4d12;
    border: 1px solid #efc78f;
    border-radius: 10px;
    padding: 0.15rem 0.4rem;
    font-size: 0.88em;
    font-weight: 700;
}

[data-testid="stMarkdownContainer"] pre {
    background: linear-gradient(180deg, #fff8ee 0%, #f9ebd6 100%);
    border: 1px solid #ead0a7;
    border-radius: 16px;
    padding: 0.9rem 1rem;
    box-shadow: 0 10px 20px rgba(120, 78, 30, 0.08);
}

[data-testid="stMarkdownContainer"] pre code {
    background: transparent;
    border: 0;
    color: #6a3b14;
    padding: 0;
    font-weight: 600;
}

div[data-testid="stTextArea"] textarea {
    background: #fffaf2;
    color: #2f2418;
    -webkit-text-fill-color: #2f2418;
}

div[data-testid="stAudioInput"] {
    margin-bottom: 0.05rem;
}

div[data-testid="stAudioInput"] label {
    margin-bottom: 0;
}

div[data-testid="stAudioInput"] > div {
    padding-top: 0;
    padding-bottom: 0.1rem;
}

div[data-testid="stAudioInput"] audio {
    display: none;
}

.voice-reply-heading {
    margin-top: 0.2rem;
}

.voice-top-avatar-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 2.2rem;
    width: 100%;
}

.voice-top-button div[data-testid="stButton"] > button {
    min-height: 2.55rem;
    height: 2.55rem;
    width: 100%;
    padding-top: 0.35rem;
    padding-bottom: 0.35rem;
}

.voice-top-select,
.voice-top-record,
.voice-top-button {
    width: 100%;
}

.voice-top-select div[data-baseweb="select"] > div {
    min-height: 2.55rem;
    height: 2.55rem;
    width: 100%;
}

.voice-top-record div[data-testid="stAudioInput"] {
    margin-bottom: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    display: flex !important;
    justify-content: center !important;
}

.voice-top-record div[data-testid="stAudioInput"] > div {
    width: 3.35rem !important;
    max-width: 3.35rem !important;
    min-width: 3.35rem !important;
}

.voice-top-record {
    margin-top: -0.45rem;
}

.voice-top-record div[data-testid="stAudioInput"] button,
.voice-top-record div[data-testid="stAudioInput"] [data-testid="stBaseButton-secondary"],
.voice-top-record div[data-testid="stAudioInput"] [data-baseweb="button"],
.voice-top-record div[data-testid="stAudioInput"] [role="button"] {
    min-height: 3.35rem !important;
    height: 3.35rem !important;
    width: 3.35rem !important;
    min-width: 3.35rem !important;
    max-width: 3.35rem !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    border-radius: 999px !important;
    box-sizing: border-box !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: linear-gradient(180deg, #fff3df 0%, #f0b35a 100%) !important;
    border: 2px solid #9a4d12 !important;
    color: #7a3510 !important;
    box-shadow: 0 6px 16px rgba(154, 77, 18, 0.2) !important;
}

.voice-top-record div[data-testid="stAudioInput"] button:hover,
.voice-top-record div[data-testid="stAudioInput"] [data-testid="stBaseButton-secondary"]:hover,
.voice-top-record div[data-testid="stAudioInput"] [data-baseweb="button"]:hover,
.voice-top-record div[data-testid="stAudioInput"] [role="button"]:hover {
    background: linear-gradient(180deg, #ffe8c4 0%, #e69a34 100%) !important;
    border-color: #7a3510 !important;
}

.voice-top-record div[data-testid="stAudioInput"] svg,
.voice-top-record div[data-testid="stAudioInput"] button span,
.voice-top-record div[data-testid="stAudioInput"] [role="button"] span {
    color: #7a3510 !important;
    fill: #7a3510 !important;
}

.voice-top-record div[data-testid="stAudioInput"] > div {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

.voice-top-record button {
    min-height: 3.35rem !important;
    height: 3.35rem !important;
    width: 3.35rem !important;
    min-width: 3.35rem !important;
    max-width: 3.35rem !important;
    border-radius: 999px !important;
    box-sizing: border-box !important;
}

.voice-mini-actions div[data-testid="stButton"] > button {
    min-height: 2rem;
    height: 2rem;
    padding: 0.15rem 0.35rem;
    font-size: 0.78rem;
    border-radius: 10px;
    font-weight: 800;
}

.auth-watermark {
    position: fixed;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 6rem;
    font-weight: 800;
    color: rgba(94, 53, 20, 0.08);
    letter-spacing: 0.08em;
    pointer-events: none;
    user-select: none;
    z-index: 998;
    transform: rotate(-22deg);
}

.auth-overlay {
    position: fixed;
    inset: 0;
    background: rgba(248, 243, 234, 0.74);
    backdrop-filter: blur(4px);
    z-index: 997;
}

.auth-card {
    position: relative;
    z-index: 999;
    max-width: 28rem;
    margin: 10vh auto 0 auto;
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #e8dcc7;
    border-radius: 24px;
    padding: 1.2rem 1.1rem;
    box-shadow: 0 18px 34px rgba(89, 63, 26, 0.12);
}

div[data-testid="stTextInput"],
div[data-testid="stButton"],
div[data-testid="stAlert"],
div[data-testid="stCaptionContainer"] {
    position: relative;
    z-index: 999;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"],
[data-testid="stText"] {
    color: #2f2418;
}

[data-testid="stCaptionContainer"] {
    font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)

if "is_authenticated" not in st.session_state:
    st.session_state["is_authenticated"] = False
if "login_error" not in st.session_state:
    st.session_state["login_error"] = ""

if not st.session_state.get("is_authenticated"):
    st.markdown('<div class="auth-overlay"></div><div class="auth-watermark">AI CHEF</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="auth-card">
    <div class="card-title">Login Required</div>
    <div class="card-copy">Enter your user ID and password to continue to Indian Vegeterian AI Chef.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    auth_left, auth_center, auth_right = st.columns([1.2, 1.0, 1.2])
    with auth_center:
        st.text_input("User ID", key="login_username", autocomplete="off")
        st.text_input("Password", key="login_password", type="password", autocomplete="off")
        st.button(
            "Login",
            use_container_width=True,
            on_click=attempt_login,
            disabled=not (APP_USERNAME and APP_PASSWORD),
        )
        if st.session_state.get("login_error"):
            st.error(st.session_state["login_error"])
        if not (APP_USERNAME and APP_PASSWORD):
            st.warning("Login credentials are not configured. Set APP_USERNAME and APP_PASSWORD in .env or Streamlit secrets.")
    st.stop()

if "agent" not in st.session_state:
    st.session_state.agent = ChefAgent()
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)
if "app_initialized" not in st.session_state:
    st.session_state["app_initialized"] = True
    st.session_state["chat_history"] = []
    st.session_state["voice_history"] = []
    st.session_state["pending_prompt"] = ""
    st.session_state["pending_text_query"] = ""
    st.session_state["pending_text_add_to_chat"] = False
    st.session_state["voice_input_value"] = ""
    st.session_state["voice_question_selection"] = ""
    st.session_state["text_question_selection"] = ""
    st.session_state["ask_chef_input"] = ""
    clear_latest_response()
    st.session_state["voice_last_question"] = ""
    st.session_state["voice_last_answer"] = ""
    st.session_state["voice_last_error"] = ""
    st.session_state["voice_status_message"] = "Ready to record a recipe question."
    st.session_state["last_voice_audio_hash"] = ""
    st.session_state["selected_prompt"] = PROMPT_PLACEHOLDER
    st.session_state["auto_speak_pending"] = False
    st.session_state["speak_text_once"] = ""
    st.session_state["last_spoken_text"] = ""
    st.session_state["last_spoken_audio_b64"] = ""
    st.session_state["voice_widget_reset"] = 0
    st.session_state["voice_language"] = "English"
    st.session_state["last_spoken_signature"] = ""
    st.session_state["pending_stop_voice"] = False
    st.session_state["experience_mode"] = "Text Assistant"

mode_param = normalize_query_param(st.query_params.get("mode"))
chef_session_flag = normalize_query_param(st.query_params.get("chef_session"))
voice_query = normalize_query_param(st.query_params.get("voice_query"))
if mode_param == "voice" or voice_query:
    st.session_state["experience_mode"] = "Voice Control"

if chef_session_flag == "1":
    st.session_state["pending_prompt"] = (
        "Introduce yourself as AI Chef and ask how you can help with recipes today."
    )
    st.session_state["auto_speak_pending"] = True
    st.query_params.clear()
elif voice_query:
    st.session_state["pending_prompt"] = voice_query
    st.session_state["voice_input_value"] = voice_query
    st.session_state["auto_speak_pending"] = True
    st.query_params.clear()

st.markdown(
    """
<div class="hero-shell">
    <div class="hero-main">
        <div class="hero-title">Indian Vegeterian AI Chef</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

mode_left_spacer, mode_center_col, mode_right_spacer = st.columns([1.2, 1.6, 1.2])
with mode_center_col:
    mode = st.radio(
        "Experience",
        ["Text Assistant", "Voice Control", "Usage"],
        horizontal=True,
        label_visibility="collapsed",
        key="experience_mode",
    )

pending_prompt_error = None
if mode == "Voice Control":
    pending_prompt_error = process_pending_prompt(
        add_to_chat=True,
        history_key="voice_history",
        source="voice",
    )

def render_latest_answer(preferred_history_key=None):
    latest_question, latest_answer = get_latest_exchange(preferred_history_key)

    st.markdown('<div class="section-heading">Latest Answer</div>', unsafe_allow_html=True)
    if latest_answer:
        if latest_question:
            st.markdown(
                f'<div class="latest-question-text">Latest Question: {latest_question}</div>',
                unsafe_allow_html=True,
            )
        st.write(latest_answer)
    else:
        st.info("Ask Chef and the latest answer will appear here.")


def render_voice_controls(show_answers=False):
    st.markdown(
        """
<div class="side-card">
    <div class="card-title">Voice Controls</div>
</div>
""",
        unsafe_allow_html=True,
    )
    voice_widget_reset = st.session_state.get("voice_widget_reset", 0)
    audio_input_key = f"voice_audio_capture_{voice_widget_reset}"
    st.button("New Chat", use_container_width=True, on_click=clear_chat)
    st.markdown("")
    st.markdown("**Voice Language**")
    st.selectbox(
        "Voice Language",
        list(VOICE_LANGUAGES.keys()),
        key="voice_language",
        help="Choose the language for voice transcription and spoken replies.",
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="voice-help-text">Click speaker to record your question.and push the red button to complete</div>',
        unsafe_allow_html=True,
    )
    voice_clip = st.audio_input(
        "Record your recipe question",
        key=audio_input_key,
        help="Use the built-in recorder to start and stop your voice question.",
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div class="voice-status-pill"><span class="voice-dot"></span><span>{st.session_state.get("voice_status_message", "Ready to record a recipe question.")}</span></div>',
        unsafe_allow_html=True,
    )
    if voice_clip is not None:
        clip_hash = hashlib.sha1(voice_clip.getvalue()).hexdigest()
        if clip_hash != st.session_state.get("last_voice_audio_hash", ""):
            try:
                st.session_state["voice_status_message"] = "Transcribing your recording..."
                transcript, transcription_error = transcribe_voice_clip(voice_clip)
                st.session_state["last_voice_audio_hash"] = clip_hash
                if transcription_error:
                    st.session_state["voice_last_question"] = ""
                    st.session_state["voice_last_answer"] = ""
                    st.session_state["voice_last_error"] = VOICE_QUERY_GUIDANCE_MESSAGE
                    st.session_state["voice_status_message"] = VOICE_QUERY_GUIDANCE_MESSAGE
                elif transcript:
                    st.session_state["voice_input_value"] = transcript
                    st.session_state["voice_status_message"] = "Generating AI Chef reply..."
                    _, error_message = run_agent_query(
                        transcript,
                        add_to_chat=True,
                        history_key="voice_history",
                        source="voice",
                    )
                    if error_message:
                        st.session_state["voice_last_error"] = error_message
                        st.session_state["voice_status_message"] = "Question not understood, ask about any Indian vegeterian recipies"
                    elif st.session_state.get("voice_latest_answer"):
                        st.session_state["speak_text_once"] = st.session_state["voice_latest_answer"]
                        st.session_state["voice_last_error"] = ""
                        st.session_state["voice_status_message"] = ""
                    st.rerun()
                else:
                    st.session_state["voice_last_question"] = ""
                    st.session_state["voice_last_answer"] = ""
                    st.session_state["voice_last_error"] = VOICE_QUERY_GUIDANCE_MESSAGE
                    st.session_state["voice_status_message"] = VOICE_QUERY_GUIDANCE_MESSAGE
            except Exception as exc:
                st.session_state["voice_last_question"] = ""
                st.session_state["voice_last_answer"] = ""
                st.session_state["voice_last_error"] = VOICE_QUERY_GUIDANCE_MESSAGE
                st.session_state["voice_status_message"] = VOICE_QUERY_GUIDANCE_MESSAGE

    query = st.session_state.get("pending_prompt", "").strip()

    if query:
        try:
            st.session_state["voice_status_message"] = "Generating AI Chef reply..."
            _, error_message = run_agent_query(
                query,
                add_to_chat=True,
                history_key="voice_history",
                source="voice",
            )
            if error_message:
                st.session_state["voice_last_error"] = error_message
                st.session_state["voice_status_message"] = "I received your request, but could not answer it."
                st.warning(error_message)
            else:
                st.session_state["voice_status_message"] = "Reply ready."
        except Exception as exc:
            st.session_state["voice_last_question"] = query
            st.session_state["voice_last_answer"] = ""
            st.session_state["voice_last_error"] = f"Could not generate a response: {exc}"
            st.session_state["voice_status_message"] = "I hit an error while answering your request."
            st.error(f"Could not generate a response: {exc}")

    if pending_prompt_error:
        st.session_state["voice_last_error"] = pending_prompt_error
        st.session_state["voice_status_message"] = "I hit an error while finishing the voice request."
        st.error(pending_prompt_error)

    voice_latest_question = st.session_state.get("voice_last_question", "")
    voice_latest_answer = st.session_state.get("voice_last_answer", "")
    voice_last_error = st.session_state.get("voice_last_error", "")
    st.markdown('<div class="voice-mini-actions">', unsafe_allow_html=True)
    replay_col, stop_col, clear_col = st.columns(3, gap="small")
    with replay_col:
        if st.button("Replay", key="start_voice_reply", use_container_width=True, disabled=not voice_latest_answer):
            st.session_state["speak_text_once"] = voice_latest_answer
            st.session_state["voice_status_message"] = "Replaying the latest spoken reply."
    with stop_col:
        if st.button("Stop", key="stop_voice_reply_top", use_container_width=True):
            stop_speaking()
            st.session_state["voice_status_message"] = "Voice playback stopped."
    with clear_col:
        if st.button("Clear", key="clear_voice_reply_top", use_container_width=True):
            stop_speaking()
            clear_chat()
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_voice_avatar():
    st.markdown(
        """
<div class="voice-top-avatar-wrap">
    <div class="agent-avatar-card voice-avatar-card">
        <div class="hero-avatar" id="chef-avatar-voice">
            <div class="chef-hat"></div>
            <div class="chef-face">
                <div class="chef-eye left"></div>
                <div class="chef-eye right"></div>
                <div class="chef-smile"></div>
            </div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_voice_answer_panel():
    voice_latest_question = st.session_state.get("voice_last_question", "")
    voice_latest_answer = st.session_state.get("voice_last_answer", "")
    voice_last_error = st.session_state.get("voice_last_error", "")

    st.markdown('<div class="section-heading">Latest Answer</div>', unsafe_allow_html=True)
    if voice_last_error:
        st.error(voice_last_error)
    elif voice_latest_answer:
        if voice_latest_question:
            st.markdown(
                f'<div class="latest-question-text">Latest Voice Question: {voice_latest_question}</div>',
                unsafe_allow_html=True,
            )
        st.write(voice_latest_answer)
    elif voice_latest_question:
        st.info("I heard your question, but I do not have a response yet.")
    else:
        st.info("Record a recipe question and the latest answer will appear here.")


def play_pending_speech():
    if not st.session_state.get("speak_text_once"):
        return

    try:
        speak_text(st.session_state["speak_text_once"])
    except Exception as exc:
        st.session_state["voice_status_message"] = f"Voice playback failed: {exc}"
    finally:
        st.session_state["speak_text_once"] = ""


if mode == "Text Assistant":
    left_col, center_col, right_col = st.columns([0.72, 1.7, 0.78], gap="medium")
    text_generation_message = ""
    text_generation_level = ""

    with left_col:
        st.markdown(
            """
<div class="side-card">
    <div class="card-title">Today's Special</div>
    <div class="card-copy">
        Load a ready recipe into the latest answer panel, or use a quick prompt below.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        recipe_of_day_clicked = st.button("Recipe Of The Day", use_container_width=True)
        st.markdown("")
        st.markdown('<div class="prompt-label">Quick Prompts</div>', unsafe_allow_html=True)
        st.selectbox(
            "Quick Prompts",
            [PROMPT_PLACEHOLDER] + SUGGESTED_PROMPTS,
            key="selected_prompt",
            label_visibility="collapsed",
            on_change=apply_selected_prompt,
        )

        if recipe_of_day_clicked:
            show_recipe_of_the_day()

    with right_col:
        st.button("New Chat", use_container_width=True, on_click=clear_chat)
        st.markdown("")
        st.markdown('<div class="section-heading">Interactive Agent</div>', unsafe_allow_html=True)
        pending_prompt = st.session_state.get("pending_prompt", "").strip()
        typed_prompt = st.chat_input("Message AI Chef", key="text_assistant_chat_input")
        query = typed_prompt or pending_prompt

        if query:
            queue_text_query(query, add_to_chat=True)
            st.session_state["pending_prompt"] = ""

        chat_messages = [
            exchange
            for exchange in st.session_state["chat_history"]
            if exchange.get("source") == "chat"
        ]
        if chat_messages:
            for exchange in chat_messages:
                with st.chat_message("user"):
                    st.write(exchange["question"])
        else:
            st.info("Continue with follow-up text questions here.")

        text_question_options = get_text_question_options()
        st.markdown('<div class="prompt-label">Session Questions</div>', unsafe_allow_html=True)
        if text_question_options:
            current_selection = st.session_state.get("text_question_selection", "")
            if current_selection not in text_question_options:
                st.session_state["text_question_selection"] = text_question_options[0]
            st.selectbox(
                "Session Questions",
                text_question_options,
                key="text_question_selection",
                label_visibility="collapsed",
            )
        else:
            st.info("Your session questions will appear here with the latest one on top.")

    with center_col:
        pending_text_query = st.session_state.get("pending_text_query", "").strip()
        pending_text_add_to_chat = st.session_state.get("pending_text_add_to_chat", False)

        if pending_text_query:
            with st.spinner("AI Chef is cooking up a response..."):
                try:
                    _, error_message = run_agent_query(
                        pending_text_query,
                        add_to_chat=pending_text_add_to_chat,
                    )
                    if error_message:
                        text_generation_message = error_message
                        text_generation_level = "warning"
                except Exception as exc:
                    text_generation_message = f"Could not generate a response: {exc}"
                    text_generation_level = "error"
                finally:
                    st.session_state["pending_text_query"] = ""
                    st.session_state["pending_text_add_to_chat"] = False

        if text_generation_level == "warning" and text_generation_message:
            st.warning(text_generation_message)
        elif text_generation_level == "error" and text_generation_message:
            st.error(text_generation_message)

        render_latest_answer()

elif mode == "Voice Control":
    voice_left_col, voice_center_col, voice_right_col = st.columns([0.72, 1.7, 0.78], gap="medium")
    with voice_left_col:
        render_voice_controls(show_answers=True)
    with voice_center_col:
        render_voice_avatar()
        render_voice_answer_panel()
        process_pending_voice_stop()
        play_pending_speech()
    with voice_right_col:
        current_question, current_answer = get_latest_exchange("voice_history")
        if current_answer:
            render_recipe_visuals(current_question, current_answer, source="voice")
        else:
            st.markdown(
                """
<div class="side-card">
    <div class="card-title">Recipe Image</div>
    <div class="card-copy">
        The recipe image will appear here after AI Chef answers your voice question.
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
else:
    usage_left_spacer, usage_center_col, usage_right_spacer = st.columns([0.9, 1.7, 0.9])
    with usage_center_col:
        st.markdown(
            """
<div class="side-card">
    <div class="card-title">Usage</div>
    <div class="card-copy">
        AI Chef helps you discover Indian vegetarian recipes, adjust them to your taste, and explore them by text or voice.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-heading">Text Assistant</div>', unsafe_allow_html=True)
        st.markdown(
            """
Use this mode when you want to type recipe questions and keep refining the same dish.

1. Click `Recipe Of The Day` for a ready-made sample recipe.
2. Use `Quick Prompts` if you want a fast starter question.
3. Type into `Message AI Chef` in the `Interactive Agent` panel for custom requests.
4. Ask follow-up changes like `make it sweeter`, `make it spicier`, or `make it for 2 people`.
5. Read the result in `Latest Answer`.
""".strip()
        )
        st.markdown("")
        st.markdown('<div class="section-heading">Voice Control</div>', unsafe_allow_html=True)
        st.markdown(
            """
Use this mode when you want to speak your question and hear the response aloud.

1. Choose your preferred `Voice Language`.
2. Record your recipe question using the voice recorder.
3. Wait for AI Chef to transcribe, answer, and speak back.
4. Use `Replay Voice Reply` to hear it again.
5. Use `Stop Voice` to stop playback.
6. Use the question dropdown to revisit questions from the current voice session.
""".strip()
        )
        st.markdown("")
        st.markdown('<div class="section-heading">Follow-Up Questions</div>', unsafe_allow_html=True)
        st.markdown(
            """
AI Chef keeps the current recipe in context during a session.

Examples:
- `Give me a sambar recipe`
- `Make it sweeter`
- `Now make it less spicy`
- `Can you make it healthier`

These follow-ups are applied to the same recipe instead of starting over.
""".strip()
        )
        st.markdown("")
        st.markdown('<div class="section-heading">Reset Options</div>', unsafe_allow_html=True)
        st.markdown(
            """
- `New Chat` starts a fresh session and clears the current history.
- `Clear` in voice mode works like `New Chat`.
- Start a new session whenever you want to switch to a completely different recipe.
""".strip()
        )

if mode != "Voice Control":
    process_pending_voice_stop()
