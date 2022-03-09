from stanza.server import CoreNLPClient
import os

os.environ["CORENLP_HOME"] = "CORENLP"

texts = ["Hello how do you do?", "what have I become?"]

with CoreNLPClient(annotators=['tokenize', 'ssplit'], timeout=30000, memory='16G') as client:
    ann = client.annotate(texts)
