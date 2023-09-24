from django.shortcuts import render

# Create your views here.
def bootstrap(request):
    return render(request,'bootsrap.html')
def proj1(request):
    import pandas as pd

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
    from tensorflow.keras.layers import Embedding
    df = pd.read_csv("C:\\Users\\User\\Desktop\\senti analysis\\my_project\\Tweets.csv")
    tweet_df = df[['text','airline_sentiment']]
    tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
    sentiment_label = tweet_df.airline_sentiment.factorize()
        
    tweet = tweet_df.text.values
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(tweet)
    vocab_size = len(tokenizer.word_index) + 1
    encoded_docs = tokenizer.texts_to_sequences(tweet)
    padded_sequence = pad_sequences(encoded_docs, maxlen=200)

    embedding_vector_length = 32
    model = Sequential() 
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
    model.add(SpatialDropout1D(0.25))
    model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

    model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=1, batch_size=32)
    if(request.method=="POST"):
            data=request.POST
            Text=data.get('txtcmt')
            if('submit' in request.POST):
                tw = tokenizer.texts_to_sequences([Text])
                tw = pad_sequences(tw,maxlen=200)
                prediction = int(model.predict(tw).round().item())
                output=sentiment_label[1][prediction]
                return render(request,"proj1.html",context={'Text':Text,'output':output})
    return render(request,"proj1.html")