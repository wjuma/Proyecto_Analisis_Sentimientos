import tweepy  # https://github.com/tweepy/tweepy

consumer_key = "BUJohOSMFB11NeW9Wi5khkymG"
consumer_secret = "yfihvSGhHoRHSadw2KPPbYgpRx7qqJwlfTdnGIv0rixmnIXoBm"
access_token = "1262539689224962050-3Sp0lWeeeDU683EjQhX0uyBuiwVLWC"
access_token_secret = "yGePV1nDq9w8UKv1DcRbOQB4kHS99eds4dcC0Wr9IuRIV"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)  ##Puede recibir otros parametros

lista = []
ntweets = 5

def normalize(s):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s

def buscar (nombre,ntweets):
    print(nombre)
    for tweet in tweepy.Cursor(api.search,
                               q='{} -filter:retweets -filter:replies'.format(nombre),
                               lang="es",
                               #since="2020-06-06",
                               #until ="2020-06-09",
                               locale="ECU",
                               geocode="-1.95529,-78.70604,330km",
                               tweet_mode="extended").items(ntweets):
        lista.append(tweet._json["full_text"])
    print("LISTA TWEETER PRINCIPAL-->",lista)
    print('LISTA TWEETER PRINCIPAL-->',len(lista))

def limpia():
    del lista[:]
# buscar('amor')
# for i in lista:
#     print('-----------------------------------------')
#     print(i)

