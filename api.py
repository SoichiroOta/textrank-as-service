import os
import json

import responder
from textrank import TextRank


env = os.environ
DEBUG = env['DEBUG'] in ['1', 'True', 'true']
RATIO = float(env['RATIO'])
MODEL = env.get('MODEL')

api = responder.API(debug=DEBUG)
textrank = TextRank(env['LIBRARY'], MODEL)


@api.route("/")
async def get_keywords(req, resp):
    body = await req.text
    text_list = json.loads(body)
    keywords_list = [textrank.keywords(
        text, RATIO
    ) for text in text_list]
    resp_dict = dict(data=keywords_list)
    resp.media = resp_dict


if __name__ == "__main__":
    api.run()