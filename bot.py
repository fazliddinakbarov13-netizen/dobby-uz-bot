"""
Dobbi 🧙🏾‍♂️ — Expert Agentlar Boshqaruvchisi | Telegram Bot
Gemini 2.5 Flash AI bilan ishlaydi
"""

import os
import json
import asyncio
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv

import google.generativeai as genai
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ParseMode, ChatAction

# ─── Konfiguratsiya ──────────────────────────────────────────────────
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
    raise ValueError("TELEGRAM_BOT_TOKEN yoki GEMINI_API_KEY topilmadi! .env faylni tekshiring.")

# Admin ID lari (botga /myid yuborib ID ni bilib oling, keyin shu yerga qo'shing)
ADMIN_IDS_STR = os.getenv("ADMIN_IDS", "")
ADMIN_IDS: set[int] = set()
if ADMIN_IDS_STR:
    ADMIN_IDS = {int(x.strip()) for x in ADMIN_IDS_STR.split(",") if x.strip().isdigit()}

# ─── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── Foydalanuvchilar bazasi (JSON) ──────────────────────────────────
USERS_FILE = Path(__file__).parent / "users.json"


def load_users() -> dict:
    """users.json dan foydalanuvchilarni yuklash."""
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_users(users: dict) -> None:
    """Foydalanuvchilarni users.json ga saqlash."""
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logger.error(f"Users saqlashda xatolik: {e}")


def register_user(user) -> None:
    """Foydalanuvchini bazaga qo'shish/yangilash."""
    users = load_users()
    user_id = str(user.id)
    
    if user_id not in users:
        users[user_id] = {
            "id": user.id,
            "first_name": user.first_name,
            "last_name": user.last_name or "",
            "username": user.username or "",
            "joined_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "is_active": True,
        }
        logger.info(f"Yangi foydalanuvchi: {user.first_name} (ID: {user.id})")
    else:
        users[user_id]["last_active"] = datetime.now().isoformat()
        users[user_id]["first_name"] = user.first_name
        users[user_id]["is_active"] = True
    
    save_users(users)

# ─── Gemini sozlamalari ──────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)

# Mavjud eng yangi model
MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = """
Act as Dobbi 🧙🏾‍♂️, a conductor of expert agents. Your job is to support the user in accomplishing their goals by aligning with their goals and preference, then calling upon an expert agent perfectly suited to the task by initializing:

"Dobbi_COR" = "${emoji}: I am an expert in ${role}. I know ${context}. I will reason step-by-step to determine the best course of action to achieve ${goal}. I can use ${tools} to help in this process.

I will help you accomplish your goal by following these steps:
${reasoned steps}

My task ends when ${completion}.
${first step, question}."

Follow these steps:
1. 🧙🏾‍♂️, Start each interaction by gathering context, relevant information and clarifying the user's goals by asking them questions.
2. Once user has confirmed, initialize "Dobbi_COR".
3. 🧙🏾‍♂️ and the expert agent, support the user until the goal is accomplished.

Commands:
/start - introduce yourself and begin with step one
/save - restate SMART goal, summarize progress so far, and recommend a next step
/reason - Dobbi and Agent reason step by step together and make a recommendation for how the user should proceed
/settings - update goal or agent
/new - Forget previous input

Rules:
- End every output with a question or a recommended next step.
- List your commands in your first output or if the user asks.
- 🧙🏾‍♂️, ask before generating a new agent.
- ALWAYS write in Uzbek language!

IMPORTANT FORMATTING RULES:
- Do NOT use Markdown formatting (no **, no __, no `, no ```).
- Use plain text with emojis for structure.
- Use bullet points with • or dashes -.
- Use numbered lists with 1. 2. 3.
- Use CAPITAL LETTERS for emphasis instead of bold.
- Use line breaks for readability.
""".strip()

# ─── Foydalanuvchi sessiyalari ───────────────────────────────────────
user_sessions: dict[int, dict] = {}

# Rate limiting - har bir foydalanuvchi uchun
user_last_request: dict[int, float] = {}
RATE_LIMIT_SECONDS = 2  # Minimal interval between requests


def get_session(user_id: int) -> dict:
    """Foydalanuvchi sessiyasini olish yoki yaratish."""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "history": [],
            "goal": None,
            "depth": "standard",  # standard | deep | surface
            "created_at": datetime.now().isoformat(),
            "message_count": 0,
        }
    return user_sessions[user_id]


def check_rate_limit(user_id: int) -> bool:
    """Rate limitni tekshirish. True = ruxsat, False = cheklangan."""
    now = time.time()
    last = user_last_request.get(user_id, 0)
    if now - last < RATE_LIMIT_SECONDS:
        return False
    user_last_request[user_id] = now
    return True


def build_chat_history(session: dict) -> list[dict]:
    """Gemini uchun chat tarixini tuzish."""
    return session["history"]


async def ask_gemini(user_id: int, user_message: str, image_data: bytes = None) -> str:
    """Gemini API ga so'rov yuborish."""
    try:
        session = get_session(user_id)
        session["message_count"] += 1

        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_PROMPT,
        )

        # Chat tarixini tuzish
        history = build_chat_history(session)
        chat = model.start_chat(history=history)

        # Rasm bilan yoki rasmsiz so'rov
        if image_data:
            import PIL.Image
            image = PIL.Image.open(BytesIO(image_data))
            # Rasmli so'rov uchun alohida model (chat emas)
            response = await asyncio.to_thread(
                model.generate_content, [user_message, image]
            )
            answer = response.text
        else:
            # Matnli so'rov
            response = await asyncio.to_thread(chat.send_message, user_message)
            answer = response.text

        # Tarixga qo'shish
        session["history"].append({"role": "user", "parts": [user_message]})
        session["history"].append({"role": "model", "parts": [answer]})

        # Tarixni cheklash (xotira tejash - oxirgi 30 ta xabar)
        if len(session["history"]) > 30:
            session["history"] = session["history"][-30:]

        return answer

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Gemini xatosi (user {user_id}): {error_msg}")
        logger.error(traceback.format_exc())

        if "404" in error_msg:
            return "⚠️ Model topilmadi. Tizim administratoriga murojaat qiling."
        elif "429" in error_msg or "quota" in error_msg.lower():
            return "⏳ So'rovlar limiti tugadi. 1 daqiqa kutib, qaytadan urinib ko'ring."
        elif "500" in error_msg or "503" in error_msg:
            return "🔧 Server vaqtincha ishlamayapti. Biroz kutib, qaytadan urinib ko'ring."
        else:
            return f"⚠️ Xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring."


async def safe_reply(message, text: str) -> None:
    """Xavfsiz javob yuborish - Telegram uzunlik cheklovlarini hisobga olish."""
    if not text:
        text = "⚠️ Bo'sh javob qaytdi. Qaytadan urinib ko'ring."

    # Telegram 4096 belgi cheklovi
    if len(text) <= 4096:
        try:
            await message.reply_text(text)
        except Exception as e:
            logger.error(f"Reply xatosi: {e}")
            await message.reply_text("⚠️ Javobni yuborishda xatolik. Qaytadan urinib ko'ring.")
    else:
        # Uzun javoblarni paragraflar bo'yicha bo'lish
        chunks = smart_split(text, 4096)
        for chunk in chunks:
            try:
                await message.reply_text(chunk)
                await asyncio.sleep(0.5)  # Flood control
            except Exception as e:
                logger.error(f"Chunk reply xatosi: {e}")


def smart_split(text: str, max_length: int) -> list[str]:
    """Matnni oqilona bo'laklarga bo'lish (paragraflar bo'yicha)."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    current = ""

    paragraphs = text.split("\n\n")
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_length:
            current += ("\n\n" + para) if current else para
        else:
            if current:
                chunks.append(current)
            # Agar bitta paragraf ham sig'masa
            if len(para) > max_length:
                for i in range(0, len(para), max_length):
                    chunks.append(para[i:i + max_length])
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks if chunks else [text[:max_length]]


# ─── Telegram komandalar ─────────────────────────────────────────────

async def notify_admins(context: ContextTypes.DEFAULT_TYPE, user, action: str) -> None:
    """Adminga foydalanuvchi harakati haqida qisqacha xabar berish."""
    if not ADMIN_IDS:
        return
    if user.id in ADMIN_IDS:
        return  # Adminning o'ziga yubormaymiz

    username = f"@{user.username}" if user.username else "Mavjud emas"
    msg = f"🔔 FOYDALANUVCHI FAOLLIGI\n\n👤 Foydalanuvchi: {user.first_name}\n🔗 Username: {username}\n🆔 ID: {user.id}\n⚡ Harakat: {action}"
    
    for admin_id in ADMIN_IDS:
        try:
            await context.bot.send_message(chat_id=admin_id, text=msg)
        except Exception:
            pass  # Xatolik bo'lsa indamaymiz (masalan, admin botni bloklagan bo'lsa)



async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start — Dobby bilan tanishish."""
    user = update.effective_user
    user_id = user.id

    # Sessiyani yangilash
    if user_id in user_sessions:
        del user_sessions[user_id]
    get_session(user_id)

    # Inline keyboard
    keyboard = [
        [
            InlineKeyboardButton("🔍 Tahlil rejimi", callback_data="help_reason"),
            InlineKeyboardButton("⚙️ Sozlamalar", callback_data="help_settings"),
        ],
        [
            InlineKeyboardButton("📖 Yordam", callback_data="help_full"),
            InlineKeyboardButton("ℹ️ Bot haqida", callback_data="about"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    welcome = (
        f"🧙🏾‍♂️ DOBBI — EXPERT AGENTLAR BOSHQARUVCHISI\n\n"
        f"Salom, {user.first_name}! Men Dobbi — sizning maqsadlaringizga "
        f"erishishda yordam beradigan expert agentlar boshqaruvchisiman.\n\n"
        f"Mening vazifam — sizning maqsadingizni aniqlab, eng mos "
        f"expert agentni chaqirish va maqsadga yetguningizcha qo'llab-quvvatlash.\n\n"
        f"─────────────────────\n"
        f"💭 Maqsadingiz nima? Menga ayting, men sizga eng mos expertni topaman!"
    )

    # Foydalanuvchini bazaga saqlash
    register_user(user)

    # Adminlarga bildirishnoma jo'natish
    await notify_admins(context, user, "Botni boshladi (/start) 🚀")

    await update.message.reply_text(welcome, reply_markup=reply_markup)

    # Gemini dan tanishish
    await update.message.chat.send_action(ChatAction.TYPING)
    intro_prompt = (
        f"Foydalanuvchi /start bosdi. Uning ismi: {user.first_name}. "
        f"O'zingni 'Dobbi 🧙🏾‍♂️ — Expert agentlar boshqaruvchisi' sifatida qisqacha tanishtir. "
        f"Foydalanuvchining maqsadini aniqla — qanday sohada yordam kerak? "
        f"Javob 3-4 jumladan oshmasin. O'zbek tilida yoz."
    )
    gemini_response = await ask_gemini(user_id, intro_prompt)
    await safe_reply(update.message, gemini_response)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help — Batafsil yordam."""
    help_text = (
        "📖 BATAFSIL YORDAM\n\n"
        "🔹 BUYRUQLAR:\n\n"
        "/start — Dobbi bilan tanishish va maqsad belgilash\n"
        "/reason [muammo] — Dobbi va Expert Agent birga bosqichma-bosqich tahlil qiladi\n"
        "   Misol: /reason AI modellarni deploy qilish qiyin\n\n"
        "/save — SMART maqsadni qayta ifodalash va progressni saqlash\n"
        "/settings — Maqsad yoki expert agentni yangilash\n"
        "   /settings goal [maqsad] — Maqsad belgilash\n"
        "   /settings depth [surface/standard/deep] — Chuqurlik\n\n"
        "/new — Oldingi kiritishni unutib, yangi suhbat boshlash\n"
        "/stats — Suhbat statistikasi\n"
        "/export — Suhbat tarixini eksport qilish\n\n"
        "🔹 QOBILIYATLARI:\n\n"
        "📝 Matn — Maqsadingizni yozing, Dobbi mos expert topadi\n"
        "📷 Rasm — Rasm yuboring, expert agent tahlil qiladi\n"
        "🎤 Ovoz — Ovozli xabar yuboring (tez kunda)\n\n"
        "🔹 TAHLIL CHUQURLIKLARI:\n\n"
        "🟢 surface — Tez va aniq javoblar\n"
        "🟡 standard — Muvozanatli tahlil (default)\n"
        "🔴 deep — To'liq bosqichma-bosqich reasoning\n"
    )
    await update.message.reply_text(help_text)


async def cmd_save(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/save — Progressni saqlash."""
    user_id = update.effective_user.id
    session = get_session(user_id)

    if not session["history"]:
        await update.message.reply_text(
            "📭 Hali suhbat boshlanmagan. Avval menga muammo yoki maqsadingizni ayting."
        )
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    save_prompt = (
        "SMART maqsadni qayta ifodalab, hozirgacha bo'lgan progressni xulosa qil "
        "va keyingi bosqich sifatida tavsiya ber. "
        "Javobni quyidagi formatda ber:\n"
        "🎯 SMART MAQSAD:\n"
        "📊 PROGRESS XULOSASI:\n"
        "➡️ KEYINGI TAVSIYA ETILADIGAN QADAM:\n\n"
        "Markdown ishlatma, oddiy matn yoz. O'zbek tilida yoz."
    )
    response = await ask_gemini(user_id, save_prompt)
    await safe_reply(update.message, f"💾 PROGRESS SAQLANDI\n\n{response}")


async def cmd_reason(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/reason — Chuqur tahlil rejimi."""
    user_id = update.effective_user.id

    # Agar argument berilgan bo'lsa
    text = " ".join(context.args) if context.args else None

    if not text:
        await update.message.reply_text(
            "🔍 DOBBI + EXPERT AGENT TAHLILI\n\n"
            "Foydalanish: /reason [muammo tavsifi]\n\n"
            "Misol: /reason AI modellarni production ga chiqarish qiyinchiliklari\n\n"
            "Bu buyruq Dobbi va Expert Agentni birga bosqichma-bosqich fikrlashga undaydi:\n"
            "  🧙🏾‍♂️ Dobbi — Maqsadni aniqlaydi\n"
            "  🤖 Expert — Chuqur tahlil qiladi\n"
            "  💡 Birgalikda — Tavsiya beradi"
        )
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    reason_prompt = (
        f"Dobbi va Expert Agent sifatida quyidagi muammoni birga bosqichma-bosqich tahlil qilinglar:\n"
        f"Muammo: {text}\n\n"
        f"Formatni quyidagicha ber (Markdown ISHLATMA, oddiy matn yoz):\n"
        f"🧙🏾‍♂️ DOBBI KONTEKST: — Muammoning mohiyati\n"
        f"🤖 EXPERT TAHLILI: — Bosqichma-bosqich reasoning\n"
        f"💡 BIRGALIKDAGI TAVSIYA: — Qanday davom etish kerak\n"
        f"❓ KEYINGI SAVOL YOKI QADAM:"
    )
    response = await ask_gemini(user_id, reason_prompt)
    await safe_reply(update.message, response)


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/settings — Sozlamalar."""
    user_id = update.effective_user.id
    session = get_session(user_id)

    args = context.args if context.args else []

    if not args:
        current_goal = session.get("goal", "Belgilanmagan")
        current_depth = session.get("depth", "standard")

        depth_labels = {
            "surface": "🟢 Yuzaki",
            "standard": "🟡 Standart",
            "deep": "🔴 Chuqur",
        }

        keyboard = [
            [
                InlineKeyboardButton("🟢 Yuzaki", callback_data="depth_surface"),
                InlineKeyboardButton("🟡 Standart", callback_data="depth_standard"),
                InlineKeyboardButton("🔴 Chuqur", callback_data="depth_deep"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"⚙️ JORIY SOZLAMALAR\n\n"
            f"🎯 Maqsad: {current_goal}\n"
            f"📏 Tahlil chuqurligi: {depth_labels.get(current_depth, current_depth)}\n\n"
            f"O'zgartirish uchun:\n"
            f"  /settings goal [maqsad] — Maqsadni belgilash\n"
            f"  Yoki quyidagi tugmalardan chuqurlikni tanlang:",
            reply_markup=reply_markup,
        )
        return

    if args[0] == "goal" and len(args) > 1:
        goal = " ".join(args[1:])
        session["goal"] = goal
        await update.message.reply_text(f"🎯 Maqsad yangilandi: {goal}")

        # Gemini ga ham xabar berish
        await ask_gemini(
            user_id,
            f"Foydalanuvchi maqsadini yangiladi: '{goal}'. Buni eslab qol va keyingi javoblarda hisobga ol.",
        )

    elif args[0] == "depth" and len(args) > 1:
        depth = args[1].lower()
        if depth in ("surface", "standard", "deep"):
            session["depth"] = depth
            depth_labels = {
                "surface": "🟢 Yuzaki — tez va aniq javoblar",
                "standard": "🟡 Standart — muvozanatli tahlil",
                "deep": "🔴 Chuqur — to'liq 5 bosqichli framework",
            }
            await update.message.reply_text(f"📏 Chuqurlik yangilandi: {depth_labels[depth]}")
        else:
            await update.message.reply_text(
                "❌ Noto'g'ri qiymat. surface, standard yoki deep tanlang."
            )
    else:
        await update.message.reply_text("❌ Noto'g'ri buyruq. /settings ni parametrsiz yuboring.")


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/new — Kontekstni tozalash."""
    user_id = update.effective_user.id

    if user_id in user_sessions:
        del user_sessions[user_id]

    await update.message.reply_text(
        "🔄 OLDINGI KIRITISH UNUTILDI!\n\n"
        "Yangi suhbatga tayyormiz. Dobbi 🧙🏾‍♂️ sizni kutmoqda.\n\n"
        "💭 Yangi maqsadingiz nima? Menga ayting, mos expert topaman!"
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/stats — Suhbat statistikasi."""
    user_id = update.effective_user.id
    session = get_session(user_id)

    msg_count = session.get("message_count", 0)
    history_len = len(session.get("history", []))
    goal = session.get("goal", "Belgilanmagan")
    depth = session.get("depth", "standard")
    created = session.get("created_at", "Noma'lum")

    depth_labels = {
        "surface": "🟢 Yuzaki",
        "standard": "🟡 Standart",
        "deep": "🔴 Chuqur",
    }

    await update.message.reply_text(
        f"📊 SUHBAT STATISTIKASI\n\n"
        f"💬 Jami xabarlar: {msg_count}\n"
        f"📝 Tarix hajmi: {history_len // 2} juft\n"
        f"🎯 Maqsad: {goal}\n"
        f"📏 Chuqurlik: {depth_labels.get(depth, depth)}\n"
        f"🕐 Boshlangan: {created[:19]}\n"
        f"🤖 Model: {MODEL_NAME}"
    )


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/export — Suhbat tarixini eksport qilish."""
    user_id = update.effective_user.id
    session = get_session(user_id)

    if not session["history"]:
        await update.message.reply_text("📭 Eksport qilish uchun suhbat tarixi yo'q.")
        return

    # Tarixni matn formatiga o'tkazish
    export_text = f"DOBBY SUHBAT TARIXI\n"
    export_text += f"Sana: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    export_text += f"Maqsad: {session.get('goal', 'Belgilanmagan')}\n"
    export_text += "=" * 40 + "\n\n"

    for i, msg in enumerate(session["history"]):
        role = "👤 FOYDALANUVCHI" if msg["role"] == "user" else "🤖 DOBBY"
        parts_text = msg["parts"][0] if isinstance(msg["parts"][0], str) else "[media]"
        export_text += f"{role}:\n{parts_text}\n\n---\n\n"

    # Fayl sifatida yuborish
    file_bytes = export_text.encode("utf-8")
    from io import BytesIO
    file_io = BytesIO(file_bytes)
    file_io.name = f"dobby_export_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"

    await update.message.reply_document(
        document=file_io,
        caption="📄 Suhbat tarixi eksport qilindi!"
    )


# ─── Callback querylar (inline tugmalar) ─────────────────────────────


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Inline tugmalar uchun callback handler."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    data = query.data

    if data.startswith("depth_"):
        depth = data.replace("depth_", "")
        session = get_session(user_id)
        session["depth"] = depth
        depth_labels = {
            "surface": "🟢 Yuzaki — tez va aniq javoblar",
            "standard": "🟡 Standart — muvozanatli tahlil",
            "deep": "🔴 Chuqur — to'liq 5 bosqichli framework",
        }
        await query.edit_message_text(f"📏 Chuqurlik yangilandi: {depth_labels[depth]}")

    elif data == "help_reason":
        await query.edit_message_text(
            "🔍 DOBBI + EXPERT TAHLIL REJIMI\n\n"
            "/reason [muammo] buyrug'i — Dobbi va Expert Agent birga tahlil qiladi:\n\n"
            "🧙🏾‍♂️ Dobbi — Kontekstni aniqlaydi\n"
            "🤖 Expert — Bosqichma-bosqich tahlil\n"
            "💡 Natija — Birgalikda tavsiya\n\n"
            "Misol: /reason Nima uchun startaplar muvaffaqiyatsiz bo'ladi"
        )

    elif data == "help_settings":
        await query.edit_message_text(
            "⚙️ SOZLAMALAR\n\n"
            "/settings — Joriy sozlamalarni ko'rish\n"
            "/settings goal [maqsad] — Maqsad belgilash\n"
            "/settings depth [surface|standard|deep] — Chuqurlik"
        )

    elif data == "help_full":
        await query.edit_message_text(
            "📖 BUYRUQLAR RO'YXATI\n\n"
            "/start — Qayta boshlash\n"
            "/reason [muammo] — Chuqur tahlil\n"
            "/save — Progress saqlash\n"
            "/settings — Sozlamalar\n"
            "/new — Yangi suhbat\n"
            "/stats — Statistika\n"
            "/export — Tarixni eksport\n"
            "/help — Batafsil yordam\n\n"
            "📷 Rasm yuborib tahlil qildiring!\n"
            "💬 Oddiy xabar yozing — Dobby javob beradi!"
        )

    elif data == "about":
        await query.edit_message_text(
            "ℹ️ DOBBI HAQIDA\n\n"
            "Dobbi 🧙🏾‍♂️ — Expert agentlar boshqaruvchisi.\n\n"
            "🧠 Qanday ishlaydi:\n"
            "  • Sizning maqsadingizni aniqlaydi\n"
            "  • Eng mos expert agentni chaqiradi\n"
            "  • Maqsadga yetguningizcha qo'llab-quvvatlaydi\n\n"
            f"🤖 Model: {MODEL_NAME}\n"
            "🔧 Framework: python-telegram-bot + Google Gemini AI"
        )


# ─── Oddiy xabarlar ─────────────────────────────────────────────────


async def cmd_myid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/myid — Foydalanuvchi ID sini ko'rsatish."""
    user = update.effective_user
    register_user(user)
    await update.message.reply_text(
        f"🆔 Sizning Telegram ID: {user.id}\n\n"
        f"Admin qilish uchun .env fayliga qo'shing:\n"
        f"ADMIN_IDS={user.id}"
    )


async def cmd_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/broadcast — Barcha foydalanuvchilarga xabar yuborish (faqat admin)."""
    user_id = update.effective_user.id

    # Admin tekshirish
    if ADMIN_IDS and user_id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Bu buyruq faqat admin uchun!")
        return
    
    # Agar ADMIN_IDS bo'sh bo'lsa, birinchi ishlatgan odam admin bo'ladi
    if not ADMIN_IDS:
        ADMIN_IDS.add(user_id)
        logger.info(f"Birinchi admin belgilandi: {user_id}")

    text = " ".join(context.args) if context.args else None

    if not text:
        await update.message.reply_text(
            "📢 BROADCAST REJIMI\n\n"
            "Foydalanish: /broadcast [xabar matni]\n\n"
            "Misol: /broadcast Salom! Dobby bot yangilandi!"
        )
        return

    users = load_users()
    if not users:
        await update.message.reply_text("📭 Hali hech kim ro'yxatda yo'q.")
        return

    success = 0
    failed = 0
    blocked = 0

    await update.message.reply_text(
        f"📢 {len(users)} ta foydalanuvchiga xabar yuborilmoqda..."
    )

    for uid_str, user_data in users.items():
        if not user_data.get("is_active", True):
            continue
        try:
            await context.bot.send_message(
                chat_id=int(uid_str),
                text=f"📢 DOBBY XABARI\n\n{text}"
            )
            success += 1
            await asyncio.sleep(0.1)  # Flood control
        except Exception as e:
            error_msg = str(e)
            if "blocked" in error_msg.lower() or "deactivated" in error_msg.lower():
                blocked += 1
                # Foydalanuvchini nofaol deb belgilash
                users[uid_str]["is_active"] = False
            else:
                failed += 1
            logger.warning(f"Broadcast xatosi (user {uid_str}): {e}")

    save_users(users)  # Nofaol foydalanuvchilarni yangilash

    await update.message.reply_text(
        f"📢 BROADCAST NATIJASI\n\n"
        f"✅ Muvaffaqiyatli: {success}\n"
        f"🚫 Bloklagan: {blocked}\n"
        f"❌ Xatolik: {failed}\n"
        f"📊 Jami: {len(users)}"
    )


async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/users — Foydalanuvchilar ro'yxati (faqat admin)."""
    user_id = update.effective_user.id

    if ADMIN_IDS and user_id not in ADMIN_IDS:
        await update.message.reply_text("⛔ Bu buyruq faqat admin uchun!")
        return

    users = load_users()
    if not users:
        await update.message.reply_text("📭 Hali hech kim ro'yxatda yo'q.")
        return

    active = sum(1 for u in users.values() if u.get("is_active", True))
    inactive = len(users) - active

    text = f"👥 FOYDALANUVCHILAR\n\n"
    text += f"📊 Jami: {len(users)}\n"
    text += f"✅ Faol: {active}\n"
    text += f"🚫 Nofaol: {inactive}\n\n"
    text += "─────────────────\n"

    for uid, data in list(users.items())[:50]:  # Max 50 ta ko'rsatish
        status = "✅" if data.get("is_active", True) else "🚫"
        name = data.get("first_name", "Noma'lum")
        username = f"@{data['username']}" if data.get("username") else ""
        text += f"{status} {name} {username} (ID: {uid})\n"

    if len(users) > 50:
        text += f"\n... va yana {len(users) - 50} ta"

    await safe_reply(update.message, text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Oddiy matn xabarlarini qayta ishlash."""
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    user_text = update.message.text

    # Rate limit tekshirish
    if not check_rate_limit(user_id):
        await update.message.reply_text("⏳ Iltimos, biroz kuting...")
        return

    # Foydalanuvchini ro'yxatga olish
    register_user(update.effective_user)

    # Adminlarga bildirishnoma jo'natish
    short_text = user_text[:50] + "..." if len(user_text) > 50 else user_text
    await notify_admins(context, update.effective_user, f"Xabar yozdi: {short_text} 💬")

    # "Yozmoqda..." ko'rsatish
    await update.message.chat.send_action(ChatAction.TYPING)

    # Session depth ni hisobga olish
    session = get_session(user_id)
    depth = session.get("depth", "standard")

    if depth == "deep":
        user_text = f"[CHUQUR TAHLIL REJIMI] {user_text}\nHar bir javobda 5 bosqichli Decision Frameworkni to'liq qo'lla."
    elif depth == "surface":
        user_text = f"[TEZKOR REJIM] {user_text}\nQisqa va aniq javob ber, 2-3 jumlada."

    response = await ask_gemini(user_id, user_text)
    await safe_reply(update.message, response)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Rasm xabarlarini qayta ishlash — Gemini multimodal."""
    if not update.message or not update.message.photo:
        return

    user_id = update.effective_user.id

    # Rate limit tekshirish
    if not check_rate_limit(user_id):
        await update.message.reply_text("⏳ Iltimos, biroz kuting...")
        return

    # Foydalanuvchini ro'yxatga olish
    register_user(update.effective_user)

    # Adminlarga bildirishnoma jo'natish
    await notify_admins(context, update.effective_user, "Rasm yubordi 📷")

    await update.message.chat.send_action(ChatAction.TYPING)

    # Eng katta o'lchamdagi rasmni olish
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    # Rasmni yuklab olish
    photo_bytes = await file.download_as_bytearray()

    # Caption bormi?
    caption = update.message.caption or "Bu rasmni tahlil qil. Nimani ko'ryapsan? Qanday insight berasan?"

    response = await ask_gemini(user_id, caption, image_data=bytes(photo_bytes))
    await safe_reply(update.message, response)


# ─── Xatolarni boshqarish ───────────────────────────────────────────


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xatolarni log qilish."""
    logger.error(f"Xatolik: {context.error}", exc_info=context.error)

    if isinstance(update, Update) and update.message:
        try:
            await update.message.reply_text(
                "⚠️ Kutilmagan xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring."
            )
        except Exception:
            pass  # Javob yuborishda ham xatolik bo'lsa, skip


# ─── Bot sozlamalari ────────────────────────────────────────────────


async def post_init(application: Application) -> None:
    """Bot ishga tushganda buyruqlar ro'yxatini o'rnatish."""
    commands = [
        BotCommand("start", "Dobbi bilan tanishish"),
        BotCommand("help", "Batafsil yordam"),
        BotCommand("reason", "Dobbi + Expert tahlil"),
        BotCommand("save", "SMART maqsad va progress"),
        BotCommand("settings", "Maqsad yoki agent yangilash"),
        BotCommand("new", "Yangi suhbat boshlash"),
        BotCommand("stats", "Suhbat statistikasi"),
        BotCommand("export", "Suhbatni eksport"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info(f"✅ Dobbi bot ishga tushdi! Model: {MODEL_NAME}")


# ─── Main ────────────────────────────────────────────────────────────


def main() -> None:
    """Botni ishga tushirish."""
    # Python 3.14 uchun event loop yaratish
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(30)
        .build()
    )

    # Komandalar
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("save", cmd_save))
    app.add_handler(CommandHandler("reason", cmd_reason))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("export", cmd_export))
    app.add_handler(CommandHandler("myid", cmd_myid))
    app.add_handler(CommandHandler("broadcast", cmd_broadcast))
    app.add_handler(CommandHandler("users", cmd_users))

    # Callback querylar (inline tugmalar)
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Rasmlar
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Oddiy matn xabarlari
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Xatolar
    app.add_error_handler(error_handler)

    # Polling boshlash
    logger.info(f"🚀 Dobbi bot ishga tushmoqda... (Model: {MODEL_NAME})")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
