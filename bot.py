import logging
from textwrap import dedent

import requests
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, ConversationHandler, CallbackContext, MessageHandler, Filters, \
    PicklePersistence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = ''
QA_SERVICE_ENDPOINT = 'http://127.0.0.1:8000'
ASK_STATE = 1
SETTING_STATE = 101
SUPPORTED_ALGORITHM = ('tfidf', 'word2vec', 'doc2vec')
DEFAULT_ALGORITHM = 'tfidf'

_LIST_COMMAND = """ \
mulai      - Mulai menggunakan bot
tanya      - Bertanya tentang konstitusi
pengaturan - Mengatur algoritma
bantuan    - Menampilkan bantuan
"""


def start(update, context):
    welcome = '''\
    Selamat datang di (Under Development) Tanya Konstitusi!
    
    Tentang:
    Bot ini akan membantu anda menjawab pertanyaan tentang konstitusi Indonesia.
    
    Perintah tersedia:
    /mulai      - manampilkan pesan ini.
    /tanya      - bertanya tentang konstitusi.
    /pengaturan - mengatur algoritma.
    /bantuan    - untuk menampilkan bantuan.'''
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=dedent(welcome))


def help(update: Update, context: CallbackContext):
    help = '''\
    Bantuan
    Bot ini akan membantu anda menjawab pertanyaan tentang konstitusi Indonesia.
    
    Perintah tersedia:
    /mulai      - manampilkan pesan ini.
    /tanya      - bertanya tentang konstitusi.
    /pengaturan - mengatur algoritma.
    /bantuan    - untuk menampilkan bantuan.'''
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
    algorithm = DEFAULT_ALGORITHM
    if 'algorithm' in context.user_data \
            and context.user_data['algorithm'] in SUPPORTED_ALGORITHM:
        algorithm = context.user_data['algorithm']
    params = {'q': query}
    r = requests.get(QA_SERVICE_ENDPOINT + f'/{algorithm}', params=params)
    if r.status_code != 200:
        update.message.reply_text(
            'Sistem sedang gangguan, silahkan coba lagi nanti')
        return ConversationHandler.END

    answer = r.json()['data']['answer']

    response = f'''\
            Pertanyaan:
            {query}
            
            Jawaban:
            {answer}
            
            Algoritma digunakan: {algorithm}
            '''
    update.message.reply_text(dedent(response))

    return ConversationHandler.END


def start_setting(update: Update, context: CallbackContext) -> int:
    reply_keyboard = [SUPPORTED_ALGORITHM]
    response = '''\
        Pilih algoritma untuk Question Answering!
        
        Gunakan perintah /batal untuk membatalkan.'''
    update.message.reply_text(
        dedent(response),
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, input_field_placeholder='Algoritma?'
        ))

    return SETTING_STATE


def setting(update: Update, context: CallbackContext) -> int:
    text = update.message.text.lower()
    if text not in SUPPORTED_ALGORITHM:
        reply_keyboard = [SUPPORTED_ALGORITHM]
        response = '''\
            Algoritma tidak diketahui, silahkan pilih ulang!
            Gunakan perintah /batal untuk membatalkan.'''

        update.message.reply_text(
            dedent(response),
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, one_time_keyboard=True, input_field_placeholder='Algoritma?'
            ))
        return SETTING_STATE
    else:
        context.user_data['algorithm'] = text
        response = f'''\
                Algoritma berhasil diubah ke {text}
                '''
        update.message.reply_text(
            dedent(response),
            reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


def error(update, context):
    logger.warning(f'Update: {update}, Error: {context.error}')


def main():
    persistence = PicklePersistence(filename='.cache/data/bot')
    updater = Updater(TOKEN, persistence=persistence)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler(['start', 'mulai'], start)
    help_handler = CommandHandler(['help', 'bantuan'], help)
    ask_handler = ConversationHandler(
        entry_points=[CommandHandler(['ask', 'tanya'], start_ask)],
        states={
            ASK_STATE: [MessageHandler(Filters.text & ~Filters.command, ask)],
        },
        fallbacks=[CommandHandler(['cancel', 'batal'], cancel_ask)],
    )
    setting_handler = ConversationHandler(
        entry_points=[CommandHandler(['setting', 'pengaturan'], start_setting)],
        states={
            SETTING_STATE: [MessageHandler(Filters.text & ~Filters.command, setting)],
        },
        fallbacks=[CommandHandler(['cancel', 'batal'], cancel_ask)],
    )

    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(help_handler)
    dispatcher.add_handler(ask_handler)
    dispatcher.add_handler(setting_handler)

    dispatcher.add_error_handler(error)

    updater.start_polling()


if __name__ == "__main__":
    main()
