import logging
from textwrap import dedent

import pandas as pd
from telegram import Update
from telegram.ext import Updater, CommandHandler, ConversationHandler, CallbackContext, MessageHandler, Filters

from rc_modules import Tfidf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = ''
ASK_STATE = 1

tfidf = Tfidf(cache='.cache/tfidf')


def start(update, context):
    welcome = '''\
    Selamat datang di (Under Development) Tanya Konstitusi!
    
    Tentang:
    Bot ini akan membantu anda menjawab pertanyaan tentang konstitusi Indonesia.
    
    Perintah tersedia:
    /mulai    - manampilkan pesan ini.
    /tanya    - bertanya tentang konstitusi.
    /bantuan  - untuk menampilkan bantuan.'''
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=dedent(welcome))


def help(update: Update, context: CallbackContext):
    help = '''\
    Bantuan
    Bot ini akan membantu anda menjawab pertanyaan tentang konstitusi Indonesia.
    
    Perintah tersedia:
    /mulai    - manampilkan pesan ini.
    /tanya    - bertanya tentang konstitusi.
    /bantuan  - untuk menampilkan bantuan.'''
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=dedent(help))


def start_ask(update: Update, context: CallbackContext) -> int:
    response = '''\
        Silahkan kirimkan pertanyaan anda!

        Gunakan perintah /batal untuk membatalkan.'''
    update.message.reply_text(dedent(response))

    return ASK_STATE


def cancel_ask(update: Update, context: CallbackContext) -> int:
    update.message.reply_text('Perintah dibatalkan.')

    return ConversationHandler.END


def ask(update: Update, context: CallbackContext) -> int:
    query = update.message.text
    answer: pd.DataFrame = tfidf.ask(query=query, num_rank=1)
    answer['Response'].item()

    response = f'''\
            Pertanyaan:
            {query}
            
            Jawaban:
            {answer['Response'].item()}
            '''
    update.message.reply_text(dedent(response))

    return ConversationHandler.END


def error(update, context):
    logger.warning(f'Update: {update}, Error: {context.error}')


def main():
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler(['start', 'mulai'], start)
    help_handler = CommandHandler(['help', 'bantuan'], help)
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('tanya', start_ask)],
        states={
            ASK_STATE: [MessageHandler(Filters.text & ~Filters.command, ask)],
        },
        fallbacks=[CommandHandler(['batal', 'cancel'], cancel_ask)],
    )

    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(help_handler)
    dispatcher.add_handler(conv_handler)

    dispatcher.add_error_handler(error)

    updater.start_polling()


if __name__ == "__main__":
    main()
