"""
Created on Sat Sep 22 17:54:25 2018

@author: mush
"""

import json

import requests
import checkin

TOKEN = "643390375:AAEWalmrSz6YBmOF_bM9mkcupv8f51SSGl8"
url_base = "https://api.telegram.org/bot" + TOKEN
USER = 'Jeroen'


def __init__(self):
    self.name = 'Telegram Client'


# evaluate if the image complies with the safety regulations
def analyzeImage(img):
    # results = predictor.predict_image(project_id, img, None)

    res = {
        'status': 'nok',
        'message': 'Missing safety goggles'
    }
    return res


def setLEDstatus(status):
    headers = {'content-type': 'application/x-www-form-urlencoded'}

    if (status):
        payload = {'arg': 'off'}
    else:
        payload = {'arg': 'on'}
    requests.post('https://api.particle.io/v1/devices/26001d000b47363433353735/led?access_token'
                  '=ab3f4cb51f89c7749a1a8c6cba2db8375207e048', headers=headers, data=payload)


def debugResponse(response):
    data = json.loads(response.content)
    print("The response contains {0} properties".format(len(data)))
    for key in data:
        print(key + " : " + str(data[key]))


def findChatIdForUser(username):
    response = requests.get(url_base + '/getUpdates')
    open_chats = json.loads(response.content)
    chat_id = None

    for message in open_chats['result']:
        if message['message']['chat']['first_name'] == username:
            chat_id = message['message']['chat']['id']
            break
    return str(chat_id)


def sendPhotoToBot(image, user):
    print('Sending photo')

def sendUpdateToBot(message, user):
    # chats = requests.get(url_base + '\getUpdates', verify=True)
    print('Sending message: ' + message)
    # chat_id = findChatIdForUser(user)
    chat_id = '699337836'
    my_response = requests.post(url_base + '/sendMessage', data={'chat_id': chat_id, 'text': message})
    debugResponse(my_response)
