agent = {'User-Agent':
"Mozilla/4.0 (\
compatible;\
MSIE 6.0;\
Windows NT 5.1;\
SV1;\
.NET CLR 1.1.4322;\
.NET CLR 2.0.50727;\
.NET CLR 3.0.04506.30\
)"}


def unescape(text):
    parser = html.parser.HTMLParser()
    return (parser.unescape(text))


def translate(sent_to_translate, to_language="auto", from_language="auto"):
   
    sent_to_translate = urllib.parse.quote(sent_to_translate)
    link = "https://translate.google.com/m?hl={}&sl={}&q={}".format(to_language, from_language, sent_to_translate)
    request = urllib.request.Request(link, headers=agent)
    data = urllib.request.urlopen(request).read().decode("utf-8")
    translation = re.findall(r'class="t0">(.*?)<', data)
    if (len(translation) == 0):
        result = ''
    else:
        result = unescape(translation[0])
    return result
