

import config
import stt
import tts
from fuzzywuzzy import fuzz
import webbrowser
import random


print(f"{config.VA_NAME} (v{config.VA_VER}) начал свою работу ...")


def va_respond(voice: str):
    print(voice)
    if voice.startswith(config.VA_ALIAS):
        # обращаются к ассистенту
        cmd = recognize_cmd(filter_cmd(voice))

        if cmd['cmd'] not in config.VA_CMD_LIST.keys():
            tts.va_speak("Что?")
        else:
            execute_cmd(cmd['cmd'])


def filter_cmd(raw_voice: str):
    cmd = raw_voice

    for x in config.VA_ALIAS:
        cmd = cmd.replace(x, "").strip()

    for x in config.VA_TBR:
        cmd = cmd.replace(x, "").strip()

    return cmd


def recognize_cmd(cmd: str):
    rc = {'cmd': '', 'percent': 0}
    for c, v in config.VA_CMD_LIST.items():

        for x in v:
            vrt = fuzz.ratio(cmd, x)
            if vrt > rc['percent']:
                rc['cmd'] = c
                rc['percent'] = vrt

    return rc


def execute_cmd(cmd: str):
    if cmd == 'help':
        # help
        text = "Мен: ..."
        text += "жедел қызметтерге қоңырау шалып..."
        text += "браузерді аша аламын"
        tts.va_speak(text)
        
    elif cmd == 'SOS1':
        text = "Звоню в скорую"
        tts.va_speak(text)
    elif cmd == 'SOS1':
        text = "Звоню в полицию"
        tts.va_speak(text)
      

    elif cmd == 'joke':
        jokes = ['Как смеются программисты? ... ехе ехе ехе',
                 'ЭсКьюЭль запрос заходит в бар, подходит к двум столам и спрашивает .. «м+ожно присоединиться?»',
                 'Программист это машина для преобразования кофе в код']

        tts.va_speak(random.choice(jokes))

    elif cmd == 'open_browser':
        browser_path = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe %s'
        webbrowser.get(browser_path).open("http://python.org")


# начать прослушивание команд
stt.va_listen(va_respond)