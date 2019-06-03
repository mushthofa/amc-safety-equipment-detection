# 23-09-2019
# Azure Czure

from telegram_client import *
from checkin import *

# Change this according to work scenario
# targetClasses = ['person', 'helmet', 'safetygoggles', 'gasdetector']  # Change this according to work scenario
targetClasses = ['person', 'helmet']
targetClasses = ['person', 'helmet', 'safetygoggles']
set_targetClasses(targetClasses)
response = detect_targets()
setLEDstatus(response)
if response:
    sendUpdateToBot('Great, you are ready to go!', USER)
else:
    text = 'Warning, you are not wearing all required safety equipment for this zone.\n\n Please wear: '
    for target in targetClasses:
        if target != 'person':
            text = text + '\n\t - ' + str(target)
    sendUpdateToBot(text, USER)
