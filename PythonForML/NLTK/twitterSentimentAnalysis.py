'''
	connect the app to twitter
'''
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

# twitter app authentication stuff
ckey = '20Dmuf7Q9sRLjI4VASPpEisKg'
csecret = 'YGXcqUNl6SPUdoORH4MbiTDE2poeUoh16llSc6AWEuX51NPx8E'
atoken = '50586029-nHi9s7mQH3Wx5MrKZOkIU3iPQJ2md4ackpVOOLuOA'
asecret = 'A7fJ1yRFkABibfohcb9brvTCylc3VvC205z4Yxn6sV8G1'

# create a class inheriting from StreamListner
class listener(StreamListener):
	def on_data(self, data):
		try:
			all_data = json.loads(data)
			tweet = all_data['text']
			# find out the sentiment and confidence for teh tweet
			sentiment_value, confidence = s.sentiment(tweet)
			print(tweet, sentiment_value, confidence)
			if confidence*100 >= 80:	# 4 or more votes in favor of the sentiment
				output = open('twitter-out.txt', 'a')
				output.write(sentiment_value)
				output.write('\n')
				output.close()
			return True
		except:
			return True

	def on_error(self, status):
		print(status)

# simple app authorization with keys and token
auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
twitterStream = Stream(auth, listener())
twitterStream.filter(track=['happy'])