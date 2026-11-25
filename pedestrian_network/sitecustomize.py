import os
import urllib.request

ua = os.getenv("OX_USER_AGENT")

if ua:
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", ua)]
    urllib.request.install_opener(opener)