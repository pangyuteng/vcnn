from . import lsg
from . import train
from . import test
from . import viz
from . import train_test_reports
import yaml

import smtplib
import os 
import argparse
from path import Path
import logging

logger = logging.getLogger('utils')


def yaml2dict(file_path):
    with open(file_path,'r') as f:
        data = yaml.load(f)
    return data

def dict2yaml(file_path,info):
    with open(file_path,'w') as f:
        f.write(yaml.dump(info))
       
def txt2str(fname):
    with open(fname,'r') as f:
        r = f.read()
    return r
    
import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate

def send_email(send_to=None,title=None,content=None,user=None,
    pwd=None,smtp=None,append=None):
    
    assert isinstance(send_to, list)

    msg = MIMEMultipart(
        From=user,
        To=COMMASPACE.join(send_to),
        Date=formatdate(localtime=True),
        Subject=title,
    )    
    content ='Subject: %s\n\n' % (title) + content 
    msg.attach(MIMEText(content))       
    for _f in append:
        for k, f in _f.items():
            if k in ['plain','html']:
                with open(f, "r") as fil:         
                    msg.attach(MIMEText(fil.read(),k))
            elif k == 'attachment':
                with open(f, "rb") as fil:
                    msg.attach(MIMEApplication(
                        fil.read(),
                        Content_Disposition='attachment; filename="%s"' % basename(f),
                        Name=basename(f)
                    ))
            else:
                raise NotImplementedError()
    
    smtpserver = smtplib.SMTP(*smtp)
    smtpserver.sendmail(user, send_to, msg.as_string())
    smtpserver.quit()
    
def send_mail(content):
    logger.info('send mail...')
    from ..conf import MAIL_YML
    info = yaml2dict(MAIL_YML)
    info.update(content)
    send_email(**info)
    
    
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# def send_email_notification(to_email, to_subject, text_body):

    # msg = MIMEMultipart()
    # sender = "no_reply@mednet.ucla.edu"
    # msg["From"] = sender
    # msg["To"] = to_email
    # msg["Subject"] = to_subject
    # msg.attach(MIMEText(text_body, "plain"))
    # text = msg.as_string()

    # s = smtplib.SMTP("mail.cvib.ucla.edu")
    # s.sendmail(sender,to_email, text)
    # s.quit()
    
    
    
    
