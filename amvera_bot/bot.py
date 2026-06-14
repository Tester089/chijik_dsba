import asyncio
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import WebAppInfo, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.client.default import DefaultBotProperties

API_BASE = "https://api-hum0d0botyw2.amvera.io"
BOT_TOKEN = "8869777063:AAFmi-4d4iUcN3K2g6Ad3Znp5mI-Vl6Vb34"
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()


@dp.message(Command("start"))
async def start(message: types.Message):
    webapp = WebAppInfo(url="https://chijik-dsba-6-14.streamlit.app")
    button = InlineKeyboardButton(text="Open Web App", web_app=webapp)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[button]])
    await message.answer(
        "/store &lt;id&gt; - store history\n/stats - statistics\n/add &lt;id&gt; &lt;year&gt; &lt;month&gt; &lt;rto&gt; - add record",
        reply_markup=keyboard)


@dp.message(Command("stats"))
async def stats(message: types.Message):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_BASE}/stats", timeout=aiohttp.ClientTimeout(total=10)) as r:
            data = await r.json()
            await message.answer(
                f"<b>Dataset Statistics:</b>\n\nRecords: {data.get('total_records', 'N/A')}\nStores: {data.get('total_stores', 'N/A')}\nAvg RTO: {data.get('avg_rto', 0):.2f}\nMax: {data.get('max_rto', 0):.2f}\nMin: {data.get('min_rto', 0):.2f}")


@dp.message(Command("store"))
async def store(message: types.Message):
    store_id = int(message.text.split()[1])
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_BASE}/stores/{store_id}", timeout=aiohttp.ClientTimeout(total=10)) as r:
            data = await r.json()
            text = f"<b>Store {store_id}:</b>\n\n" + "\n".join(
                [f"{r['Месяц']}.{r['Год']}: <b>{r['РТО']}</b>" for r in data[:5]])
            await message.answer(text)


@dp.message(Command("add"))
async def add(message: types.Message):
    parts = message.text.split()
    new_id, year, month, rto = int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])
    payload = {"new_id": new_id, "Year": year, "Month": month, "РТО": rto}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_BASE}/records", json=payload, timeout=aiohttp.ClientTimeout(total=10)) as r:
            await message.answer(f"<b>Added:</b> {new_id} {month}.{year} RTO: {rto}" if r.status == 201 else "Error")


if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot, skip_updates=True))
