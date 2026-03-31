# 🏛 Dobby — Tizimlar Me'mori | Telegram Bot

Gemini 2.5 Flash AI bilan ishlaydigan Telegram bot.

## Xususiyatlari

- 🤖 **AI Suhbat** — Gemini 2.5 Flash bilan o'zbek tilida
- 📷 **Rasm tahlili** — Rasmlarni Gemini multimodal bilan tahlil qilish
- 🔍 **Chuqur tahlil** — 3 qatlamli muammo tahlili (Surface → Structure → Essence)
- 💾 **Progress saqlash** — SMART maqsadlar va compressed insight
- 📢 **Broadcast** — Barcha foydalanuvchilarga xabar yuborish
- 📊 **Statistika** — Suhbat statistikasi
- 📤 **Eksport** — Suhbat tarixini fayl sifatida yuklab olish

## Buyruqlar

| Buyruq | Tavsif |
|--------|--------|
| `/start` | Botni boshlash |
| `/help` | Batafsil yordam |
| `/reason [muammo]` | Chuqur tahlil |
| `/save` | Progressni saqlash |
| `/settings` | Sozlamalar |
| `/new` | Yangi suhbat |
| `/stats` | Statistika |
| `/export` | Tarixni eksport |
| `/myid` | Telegram ID |
| `/broadcast [xabar]` | Xabar yuborish (admin) |
| `/users` | Foydalanuvchilar (admin) |

## O'rnatish

```bash
pip install -r requirements.txt
```

## Environment Variables

```
TELEGRAM_BOT_TOKEN=your_bot_token
GEMINI_API_KEY=your_gemini_api_key
ADMIN_IDS=your_telegram_id
```

## Ishga tushirish

```bash
python bot.py
```

## Deploy (Railway)

1. GitHub ga push qiling
2. [Railway.app](https://railway.app) da yangi project yarating
3. GitHub reponi ulang
4. Environment variables qo'shing
5. Deploy!
