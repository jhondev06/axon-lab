import asyncio
import telegram

async def test_telegram():
    try:
        bot = telegram.Bot(token='8399049016:AAHskSs70ZuploRjAonJptBOlM_IG3iw4zE')
        print("Bot criado com sucesso")
        
        # Testar conexão
        updates = await bot.get_updates()
        print(f"Updates disponíveis: {len(updates)}")
        
        # Enviar mensagem
        result = await bot.send_message(
            chat_id='7980914055', 
            text='✅ Teste direto do container - Conexão funcionando!'
        )
        print(f"Mensagem enviada: {result}")
        
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    asyncio.run(test_telegram())