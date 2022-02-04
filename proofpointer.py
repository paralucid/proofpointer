import json
import keras
import h5py

from io import BytesIO
import boto3

import pickle
import os
import numpy as np
from numpy import array

from keras.models import load_model
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM

import keras.backend as K

#email import 
import boto3
from botocore.exceptions import ClientError
import re
import base64

# 16fp training
# Default is 1e-7 which is too small for float16.
# Without adjusting the epsilon, we will get NaN
# predictions because of divide by zero problems
K.set_epsilon(1e-4)
K.set_floatx('float16')

Banner = """
            __.......__
        ,-"``           ``"-.
        |;------.-'      _.-'\\
        ||______|`  ' ' `    |
        ||------|            |
       _;|______|            |_
     (```"""""""|            |``)
     \'._       '-.........-'_.'/
      '._`""===........===""`_.'

            Proof Pudding
"""

def scoreText(emailContent):
    s3_client = boto3.client('s3')

    result = s3_client.download_file("proofpointer",'texts.h5', "/tmp/texts.h5")
    result = s3_client.download_file("proofpointer",'texts.h5.vocab', "/tmp/texts.h5.vocab")
    
    print("Model loaded")

    model = load_model('/tmp/texts.h5')

    with open(f'/tmp/texts.h5.vocab', 'rb') as h:
        tokenizer = pickle.load(h)

    #text = open('test.txt').read()

    tokenized = tokenizer.texts_to_matrix([emailContent])[0]
    prediction = model.predict(np.array([tokenized]))
    prediction = int(prediction[0][0] * 1000)
    
    print(f'\n[+] Predicted Score: {prediction}\n')

    return(f'\n[+] Predicted Score: {prediction}\n')
    
def scoreLinks(links):
    s3_client = boto3.client('s3')

    result = s3_client.download_file("proofpointer",'links.h5', "/tmp/links.h5")
    result = s3_client.download_file("proofpointer",'links.h5.vocab', "/tmp/links.h5.vocab")
    
    print("Model loaded")

    model = load_model('/tmp/links.h5')

    with open(f'/tmp/links.h5.vocab', 'rb') as h:
        tokenizer = pickle.load(h)

    #text = open('test.txt').read()
    
    linkScores = {}
    
    for link in links:
        tokenized = tokenizer.texts_to_matrix([link])[0]
        prediction = model.predict(np.array([tokenized]))
        prediction = int(prediction[0][0] * 1000)
        
        #linkScores.append(f'\n{link} - [+] Predicted Score: {prediction}')
        linkScores[link] = f'[+] Predicted Score: {prediction}'
        print(f'\n[+] Predicted Score: {prediction}\n')
        
    
    return(linkScores)

def create_neural_network(input_dim, hidden_dim=64, activation='sigmoid'):
    model = Sequential()
    model.add(Dense(hidden_dim, activation=activation, input_dim=input_dim))
    model.add(Dense(hidden_dim, activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    return model

def create_neural_network_lstm(vocab_size, hidden_size, input_length, activation='sigmoid'):
    model = Sequential()
    model.add(Embedding(vocab_size, hidden_size))
    model.add(LSTM(hidden_size))
    model.add(Dense(1, activation=activation))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    return model
    
def lambda_handler(event, context):
    parsedInput = event['Records'][0]['ses']['mail']
    email = parsedInput['source']
    messageId = parsedInput['messageId']
    
    text = ''
    links = []
    
    textScore = ''
    linkScore = ''
    
    text,links = parse(messageId)
    
    if (re.match('^\s+$',text) or text == None):
    	text = '[No text sample submitted!]'
    	linkScore = scoreLinks(links)
    elif(len(links) == 0):
        textScore = scoreText(text)
        linkScore = None
    else:
    	textScore = scoreText(text)
    	linkScore = scoreLinks(links)
    
    print(text)
    send_email(email, text, textScore, linkScore)
    
def parse(messageId):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('proofpointer1')
    rawEmail = bucket.Object(key='emails/' + messageId).get()
    textEmail = str(rawEmail['Body'].read()) #preparsed? see if you can read all in.

    recipientRegex = re.compile("(?<=To:\s)(\S+)")
    subjectRegex = re.compile("(?<=Subject:\s)(.*?)(?=--_000_)")
    bodyRegex = re.compile("(?<=Content-Type: text/plain;\s)(.*?)(?=--(_000_|000))")
    b64regex = re.compile("(?<=Content-Transfer-Encoding: base64)(.*)")
    quotedRegex = re.compile("(?<=Content-Transfer-Encoding: quoted-printable)(.*)")
    urlRegex = re.compile("(https?://[\-a-zA-Z0-9]+\.[a-zA-Z0-9]{1,6}[\-a-zA-Z0-9()@:%_\+.~#?&//=]*)")
    
    print(textEmail)
    content = ''
    links = []
    
    toEmail = re.search(recipientRegex, str(textEmail))
    textArea = ''

    try: 
	    body = re.search(bodyRegex, textEmail)
	    textArea = body.group(1)

    except:
	    print("Could not parse email.")
	    return content,links
    
    if('base64' in textArea):
    	print("[+] Base64 detected")
    	try:
	    	b64 = re.search(b64regex, textArea)
	    	b64content = b64.group(1).replace('\\r\\n','')
	    	content = (base64.b64decode(b64content)).decode()
	    	content = re.sub("!|,\s*|,\n|\s*-\s+|\s*\.\s+|(?<=\w)\.(?=\s)|\.\n|\s\$(?=\w)",' ', content) #clean up punctuation
	    	content = re.sub("'", '', content)

	    	links = re.findall(urlRegex, content)
	    	content = re.sub("<.*?>",'',content)

	    	for link in links:
	    	    content = re.sub(link, '', content)

    	except:
	    	content = 'Failed to decode email!'
    
    elif('quoted-printable' in textArea):
    	print("[+] Plain text detected")

    	try:
	    	plainText = re.search(quotedRegex, textArea)
	    	content = plainText.group(1) #email MIME cleanup
	    	content = content.replace("\\r\\n", ' ')
	    	links = re.findall(urlRegex, content)
	    	
	    	content = re.sub("!|,\s*|\s*-\s+|\s*\.\s+|(?<=\w)\.+",' ', content) #clean up punctuation
	    	content = re.sub("\s+", ' ', content)
	    	content = re.sub("\s+=|=\s+|(?<=\w)=(?=\w)|(?<=/)=", '', content) #if [space]=foo OR foo=[SPACE] OR foo=bar remove the space
	    	content = re.sub("\\\'", '', content)
	    	content = content.replace('\\', '')

	    	content = re.sub("<.*?>",'',content)

	    	for link in links:
	    	    content = re.sub(link, '', content)

	    	content = re.sub('\.', '', content)
    	
    	except:
    		content = 'Failed to decode email!'
    	
    elif('charset' in textArea):
    	print("[+] Charset detected")

    	try:
	    	plainText = re.sub('charset=\".*\"', '', textArea)
	    	content = plainText.replace('\\r\\n', ' ')
	    	content = content.replace("\\'","'")                      #the model tokenizer can take care of word contractions
	    	content = re.sub("!|,\s*|\s*-\s+|\s*\.\s+",' ', content) #clean up punctuation
	    	content = re.sub("(?<=\s)=|=(?=\s)|(?<=\w)=(?=\w)|(?<=/)=|\<.*\>", '', content) #if [space]=foo OR foo=[SPACE] OR foo=bar remove the space
	    
	    	#print(content)
	    	links = re.findall(urlRegex, content)
	    	
	    	for link in links:
	            content = content.replace(link, '')

    	except:
    		content = 'Failed to decode email!'       

    
    #if all else fails
    return content,links
    
    
def send_email(recipient, content, score, links):
    
    # Replace sender@example.com with your "From" address.
    # This address must be verified with Amazon SES.
    SENDER = "Score Bot <score@phishscale.xyz>"
    
    # Replace recipient@example.com with a "To" address. If your account 
    # is still in the sandbox, this address must be verified.
    #RECIPIENT = email
    
    # Specify a configuration set. If you do not want to use a configuration
    # set, comment the following variable, and the 
    # ConfigurationSetName=CONFIGURATION_SET argument below.
    #CONFIGURATION_SET = "ConfigSet"
    
    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    AWS_REGION = "us-west-2"
    
    # The subject line for the email.
    SUBJECT = "ProofPudding Score Results"
    
    # The email body for recipients with non-HTML email clients.
    #BODY_TEXT = ("Amazon SES Test (Python)\r\n"
                #  "This email was sent with Amazon SES using the "
                #  "AWS SDK for Python (Boto)."
                # )
                
                
    # The HTML body of the email.
    
    linkOutput = ''
    if(links != None):
        for k,v in links.items():
            linkOutput += "<p>{}<br><b>{}</b></p>\n".format(k, v)
    else:
        linkOutput = 'No links found in submitted text!'
    
    BODY_HTML = """<html>
    <head></head>
    <body>
    <h3>Submitted Text:</h3>
        <p>{}</p>
        
        <p><b>{}</b></p>
        
    <h3>Submitted Links:</h3>
        {}
        
        <p>Powered by ProofPudding</p>
        
    </body>
    </html>
                """.format(content, score, linkOutput)
                
    response = """
    {}
    {}
    
    Usage tips: 
        [+] Send a new email for each score
    
    Powered by ProofPudding
    """.format(content, score)
    
    # The character encoding for the email.
    CHARSET = "UTF-8"
    
    # Create a new SES resource and specify a region.
    client = boto3.client('ses',region_name=AWS_REGION)
    
    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    recipient,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': BODY_HTML,
                    },
                    'Text': {
                        'Charset': CHARSET,
                        'Data': response,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
            # If you are not using a configuration set, comment or delete the
            # following line
            #ConfigurationSetName=CONFIGURATION_SET,
        )
    # Display an error if something goes wrong.	
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Recipient: {}".format(recipient))