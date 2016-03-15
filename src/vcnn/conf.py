import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VCNN_ROOT = os.path.join(ROOT,'src')
DATA_ROOT = os.path.join(ROOT,'data')
BIN_ROOT = os.path.join(ROOT,'bin')

CONF_PATH = os.path.join(ROOT,'conf')
MAIL_YML = os.path.join(CONF_PATH,'email.yml')