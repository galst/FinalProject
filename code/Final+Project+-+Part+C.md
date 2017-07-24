

```python
from keras.utils.data_utils import get_file
path = get_file('trump3.txt', origin="https://raw.githubusercontent.com/galst/FinalProject/master/data/trump-txt.txt")
text = open(path).read()
print('corpus length:', len(text))
```

    Downloading data from https://raw.githubusercontent.com/galst/FinalProject/master/data/trump-txt.txt
    16384/15434 [===============================] - 0s    
    corpus length: 15434
    


```python
text[0:20000]
```




    'While I greatly appreciate the efforts of President Xi & China to help w/ North Korea, it has not worked out. At least I know China tried!\nThe Democrats don’t want you to know how much we’ve accomplished. While they do nothing but obstruct and distract, we continue to work hard on fulfilling our promises to YOU!\nBig day today. VOTE Handel (GA-6) and Norman (SC-5) for Congress. These two will be a tremendous help in our fight for lower taxes, stronger security, and great healthcare!\nThe U.S. once again condemns the brutality of the North Korean regime as we mourn its latest victim.\nGREAT job Ivanka Trump!\n"President Donald J. Trump\'s schedule for Tuesday, June 20th:\nDaily intelligence briefing\nMeeting with National Security Advisor H.R. McMaster (drop-in by Vice President Mike Pence and President Petro Poroshenko of Ukraine)\nLegislative Affairs lunch\nDinner with Vice President Mike Pence and Mrs. Karen Pence"\nMy heartfelt thoughts and prayers are with the 7 U.S. Navy sailors of the #USSFitzgerald and their families.\n"Melania and I offer our deepest condolences to the family of Otto Warmbier on his untimely passing.\nThere is nothing more tragic for a parent than to lose a child in the prime of life. Our thoughts and prayers are with Otto’s family and friends, and all who loved him. \nFull Statement: whitehouse.gov/the-press-office/2017/06/19/statement-president-donald-j-trump-passing-otto-warmbier"\nThank you Wyatt and Montana — two young Americans who aren’t afraid to stand up for what they believe in. Our movement to #MAGA is working because of great people like you!\nMelania and I were honored to welcome President Juan Carlos Varela and Mrs. Varela of Panama to the White House today.\nThe MAKE AMERICA GREAT AGAIN agenda is doing very well despite the distraction of the Fake News!\n"President Donald J. Trump\'s schedule for Monday, June 19th:\nDaily intelligence briefing\nWelcomes President Juan Carlos Varela and Mrs. Varela of Panama\nMeeting with President Varela\nWorking luncheon with President Varela\nAmerican Technology Council roundtable\nAmerican Technology Council reception"\nThis is the message I want every young American to hear: there is dignity in every honest job, and there is nobility in every honest worker.\nI am canceling the last administration’s completely one-sided deal with Cuba, and seeking a MUCH better deal for the Cuban people and for the United States of America.\nNow that I am President, America will expose the crimes of the Castro regime and stand with the Cuban people in their struggle for freedom.\nAmerica will always stand for liberty, and America will always pray and cheer for the freedom of the Cuban people.\nDespite the phony witch hunt going on in America, we are doing GREAT! Regulations way down, jobs and enthusiasm way up! #MAGA\nGreat news! #MAGA\n"Thank you Wisconsin! Tuesday was a great success for #WorkforceWeek at WCTC with Ivanka Trump and Governor Scott Walker. \nMore: 45.wh.gov/BXYooL"\nThe Fake News Media hates when I use what has turned out to be my very powerful Social Media - over 100 million people! I can go around them!\n"President Donald J. Trump\'s schedule for Friday, June 16th:\nTravels to Miami, FL\nGives remarks and participates in a signing on the United States’ policy towards Cuba\nReturns to Washington, D.C."\nWe are all united by our love of our GREAT and beautiful country.\n"Today we celebrate the dignity of work and the greatness of the American worker.\nExpanding Apprenticeships in America: http://bit.ly/2stwYia"\nYou are witnessing the single greatest WITCH HUNT in American political history — led by some very bad and conflicted people!  #MAGA\n"President Donald J. Trump\'s schedule for Thursday, June 15th:\nDaily intelligence briefing\nGovernors and Workforce of Tomorrow roundtable\nRemarks on the Apprenticeship and Workforce of Tomorrow initiatives\nSigns an Executive Order\nInvestiture Ceremony for Justice Neil Gorsuch"\nHonoring our great American flag and all for which it stands. #FlagDay\nHappy birthday to the U.S. Army and our soldiers. Thank you for your bravery, sacrifices, and dedication. Proud to be your Commander-in-Chief!\nMelania and I are grateful for the heroism of our first responders and praying for the swift recovery of all victims of this terrible shooting. Please take a moment today to cherish those you love, and always remember those who serve and keep us safe.\n"President Donald J. Trump\'s statement on the shooting incident in Virginia:\nThe Vice President and I are aware of the shooting incident in Virginia and are monitoring developments closely. We are deeply saddened by this tragedy. Our thoughts and prayers are with the members of Congress, their staffs, Capitol Police, first responders, and all others affected."\nCongressman Steve Scalise of Louisiana, a true friend and patriot, was badly injured but will fully recover. Our thoughts and prayers are with him.\n"President Donald J. Trump\'s schedule for Wednesday, June 14th:\nRemarks at the Apprenticeship Initiative kickoff\nSigns an Executive Order"\nThe passage of the U.S. Department of Veterans Affairs Accountability and Whistleblower Protection Act is GREAT news for veterans! I look forward to signing it!\n2 million more people just dropped out of ObamaCare. It’s totally broken. The Obstructionist Democrats gave up, but we’re going to have a real bill, not ObamaCare!\nThe Fake News Media has never been so wrong or so dirty. Purposely incorrect stories and phony sources to meet their agenda of hate. Sad!\nHeading to the Great State of Wisconsin to talk about JOBS, JOBS, JOBS! Big progress being made as the Real News is reporting.\nThe fake news MSM doesn\'t report the great economic news we\'ve had since Election Day. All of our hard work is kicking in and we\'re seeing big results!\n"President Donald J. Trump\'s schedule for Tuesday, June 13th:\nDaily intelligence briefing\nMeeting with National Security Advisor H.R. McMaster\nLunch with members of Congress\nTravels to Milwaukee, Wisconsin\nMeeting with Obamacare victims\nStatement on healthcare\nTours Waukesha County Technical College\nWorkforce development roundtable discussion\nRemarks at a Friends of Governor Scott Walker reception\nReturns to Washington, D.C."\nFinally held our first full Cabinet meeting today. With this great team, we can restore American prosperity and bring REAL CHANGE to Washington.\nOur miners are going back to work!\nWe will NEVER FORGET the victims who lost their lives one year ago today in the horrific #PulseNightClub shooting. #OrlandoUnitedDay\n"President Donald J. Trump\'s schedule for Monday, June 12th:\nReceives National Security Council briefing\nLeads Cabinet Meeting\nLunch with Vice President Mike Pence\nWelcomes the 2016 NCAA Football National Champions: The Clemson Tigers"\nThe process to build is painfully slow, costly, and time consuming. My administration will END these terrible delays once and for all!\nAmerica is going to build again — under budget, and ahead of schedule.\nGreat honor to welcome President Klaus Iohannis to The White House today.\nI was not elected to continue a failed system. I was elected to CHANGE IT.\nDespite so many false statements and lies, total and complete vindication...and WOW, Comey is a leaker!\nCongratulations to Jeb Hensarling & Republicans on successful House vote to repeal major parts of the 2010 Dodd-Frank financial law. GROWTH!\nOne million new jobs, historic increases in military spending, and a record reduction in illegal immigration (75%). AMERICA FIRST!\n"President Donald J. Trump\'s schedule for Friday, June 9th:\nReceives daily intelligence briefing\nRoads, Rails, and Regulatory Relief Roundtable\nDelivers remarks\nBilateral meeting with the President of Romania\nExpanded bilateral meeting with the President of Romania\nJoint press conference with the President of Romania"\nWe will not let other countries take advantage of the United States anymore. We are bringing back OUR jobs!\nWe are keeping our promises — reversing government overreach and returning power back to the people, the way the country started.\nFrom now on, we will follow a very simple rule — every day that I am President, we are going to make AMERICA FIRST.\nYou fought hard for me, and now I\'m fighting hard for all of YOU.\nOur government will once again CELEBRATE and PROTECT religious freedom.\nWe will REBUILD this country because that is how we will Make America Great Again!\n"President Donald J. Trump\'s schedule for Thursday, June 8th:\nRemarks at Faith and Freedom Coalition’s Road to Majority Conference\nHosts Infrastructure Summit with Governors and Mayors"\nIt is time to rebuild OUR country, to bring back OUR jobs, to restore OUR dreams, and yes, to put #AmericaFirst!\nThe American People deserve the best infrastructure in the world!\nObamacare is in a total death spiral. We must fix this big problem so Americans can finally have the health care that they deserve!\n"Happy birthday to Vice President Mike Pence! Let\'s make his first birthday in The White House a GREAT one by signing his card. \n>>> http://bit.ly/pence-card"\nGetting ready to leave for Cincinnati, in the GREAT STATE of OHIO, to meet with ObamaCare victims and talk Healthcare & also Infrastructure!\nI will be nominating Christopher A. Wray, a man of impeccable credentials, to be the new Director of the FBI. Details to follow.\n"President Donald J. Trump\'s schedule for Wednesday, June 7th:\nReceives daily intelligence briefing\nTravels to Cincinnati, Ohio\nMeets with Obamacare victims\nRemarks on infrastructure initiative \nReturns to Washington, D.C."\nAnnouncement of Air Traffic Control Initiative:\nDuring my recent trip to the Middle East I stated that there can no longer be funding of Radical Ideology...Perhaps this will be the beginning of the END to the horror of terrorism!\nToday we remember the courage and bravery of our troops that stormed the beaches of Normandy 73 years ago. #DDay\nThe FAKE mainstream media is working so hard trying to get me not to use Social Media. I will not stop bringing my message directly to YOU, the American People!\nBig meeting today with Republican leadership concerning Tax Cuts and Healthcare. We are all pushing hard - must get it right!\n"Our nation will move faster, fly higher, and soar proudly toward the next GREAT chapter of American aviation. \n>>> http://bit.ly/2rZNOWu"\n"President Donald J. Trump\'s schedule for Tuesday, June 6th:\nReceives daily intelligence briefing\nMeets with National Security Advisor H. R. McMaster\nMeets with House and Senate leadership\nSigns a bill\nDinner with Members of Congress"\nThat\'s right, we need a TRAVEL BAN for certain DANGEROUS countries, not some politically correct term that won\'t help us protect our people!\n"Today, I announced an Air Traffic Control Initiative to take American air travel into the future - finally!\n>>>http://45.wh.gov/pmRJsy"\nSecretary Shulkin\'s decision to modernize the U.S. Department of Veterans Affairs medical records system is one of the biggest wins for our VETERANS in decades. Our HEROES deserve the best!\n"We need the Travel Ban — not the watered down, politically correct version the Justice Department submitted to the Supreme Court, but a MUCH TOUGHER version!\nWe cannot rely on the MSM to get the facts to the people. Spread this message. SHARE NOW."\nWe will do everything in our power to assist the U.K. We send our thoughts and prayers, and we renew our resolve — stronger than ever before — to protect the United States and its allies. This bloodshed MUST END.\n"President Donald J. Trump\'s schedule for Monday, June 5th:\nReceives daily intelligence briefing\nAnnounces the Air Traffic Control Reform Initiative\nLunch with Vice President Mike Pence and Secretary of Education Betsy DeVos\nHosts a reception for Gold Star Families with First Lady Melania Trump"\nWe must stop being politically correct and get down to the business of security for our people. If we don\'t get smart it will only get worse.\nWhatever the United States can do to help out in London and the U. K., we will be there - WE ARE WITH YOU. GOD BLESS!\nWe need to be smart, vigilant and tough. We need the courts to give us back our rights. We need the Travel Ban as an extra level of safety!\nToday we reaffirmed our unbreakable support for the American HEROES who keep our streets, our homes, and our citizens safe.\nThrilled to be back in the U.S. after our first foreign trip — one that was full of historic and unprecedented achievements.\nI was elected to represent the citizens of Pittsburgh, not Paris —  so we’re getting out of the Paris Accord, and we’re making America GREAT again!\nIt is time to exit the Paris Accord and time to pursue a new deal that PROTECTS the environment, our companies, our citizens, and our COUNTRY. #AmericaFirst\n"President Donald J. Trump\'s schedule for Friday, June 2nd:\nReceives daily intelligence briefing\nMeets with Senator Lindsey Graham\nSigns bills"\nIn order to fulfill my solemn duty to protect America and its citizens, the United States will withdraw from the one-sided Paris Climate Accord. We will no longer be robbed at the expense of the American worker. We are putting America FIRST!\nI will be announcing my decision on the Paris Accord, today at 3:00 P.M. in the White House Rose Garden. MAKE AMERICA GREAT AGAIN!\nThe big story is the "unmasking and surveillance" of people that took place during the Obama Administration.\n"President Donald J. Trump\'s schedule for Thursday, June 1st:\nReceives daily intelligence briefing\nMeets with National Security Advisor H. R. McMaster\nStatement regarding the Paris Accord"\nIt was an honor to welcome the Prime Minister of Vietnam, Nguy<U+1EC5>n Xuân Phúc to The White House this afternoon.\nWe traveled the world to strengthen long-standing alliances, and to form new partnerships. See more at: http://45.wh.gov/tnmVr7\nHopefully Republican Senators, good people all, can quickly get together and pass a new (repeal & replace) HEALTHCARE bill. Add saved $\'s.\nI will be announcing my decision on the Paris Accord over the next few days. MAKE AMERICA GREAT AGAIN!\nKathy Griffin should be ashamed of herself. My children, especially my 11 year old son, Barron, are having a hard time with this. Sick!\nSo now it is reported that the Democrats, who have excoriated Carter Page about Russia, don\'t want him to testify. He blows away their case against him & now wants to clear his name by showing "the false or misleading testimony by James Comey, John Brennan..." Witch Hunt!\n"President Donald J. Trump\'s schedule for Wednesday, May 31st:\nReceives daily intelligence briefing\nMeets with Secretary of State Rex Tillerson\nWelcomes Prime Minister Nguyen Xuan Phuc of Vietnam\nMeets with Nguyen Xuan Phuc\nExpanded bilateral meeting with Prime Minister Nguyen Xuan Phuc"\n"Buy American and hire American. Make sure your #MAGA hat isn\'t FAKE by watching how our hats are 100% #MadeInTheUSA!\nGet the authentic hat today: shop.donaldjtrump.com/collections/headwear"\nThe U.S. Senate should switch to 51 votes, immediately, and get Healthcare and TAX CUTS approved, fast and easy. Dems would do it, no doubt!\nRussian officials must be laughing at the U.S. & how a lame excuse for why the Dems lost the election has taken over the Fake News.\nWe have a MASSIVE trade deficit with Germany, plus they pay FAR LESS than they should on NATO & military. Very bad for U.S. This will change.\nWe honor the noblest among us -- the men and women who paid the ultimate price for victory and for freedom.'




```python
vocabulary_size = 600
unknown_token = "UNKNOWNTOKEN"
```


```python
sentence_start_token = "SENTENCESTART"
sentence_end_token = "SENTENCEEND"
```


```python
separator= "SEPARATOR"
```


```python
text1 = text.replace('\n', ' ')
text1 = text1.replace('--',' '+ separator + ' ')
text1 = text1.replace('.',' '+sentence_end_token +' '+ sentence_start_token+' ' )
text1[0:2000]
```




    'While I greatly appreciate the efforts of President Xi & China to help w/ North Korea, it has not worked out SENTENCEEND SENTENCESTART  At least I know China tried! The Democrats don’t want you to know how much we’ve accomplished SENTENCEEND SENTENCESTART  While they do nothing but obstruct and distract, we continue to work hard on fulfilling our promises to YOU! Big day today SENTENCEEND SENTENCESTART  VOTE Handel (GA-6) and Norman (SC-5) for Congress SENTENCEEND SENTENCESTART  These two will be a tremendous help in our fight for lower taxes, stronger security, and great healthcare! The U SENTENCEEND SENTENCESTART S SENTENCEEND SENTENCESTART  once again condemns the brutality of the North Korean regime as we mourn its latest victim SENTENCEEND SENTENCESTART  GREAT job Ivanka Trump! "President Donald J SENTENCEEND SENTENCESTART  Trump\'s schedule for Tuesday, June 20th: Daily intelligence briefing Meeting with National Security Advisor H SENTENCEEND SENTENCESTART R SENTENCEEND SENTENCESTART  McMaster (drop-in by Vice President Mike Pence and President Petro Poroshenko of Ukraine) Legislative Affairs lunch Dinner with Vice President Mike Pence and Mrs SENTENCEEND SENTENCESTART  Karen Pence" My heartfelt thoughts and prayers are with the 7 U SENTENCEEND SENTENCESTART S SENTENCEEND SENTENCESTART  Navy sailors of the #USSFitzgerald and their families SENTENCEEND SENTENCESTART  "Melania and I offer our deepest condolences to the family of Otto Warmbier on his untimely passing SENTENCEEND SENTENCESTART  There is nothing more tragic for a parent than to lose a child in the prime of life SENTENCEEND SENTENCESTART  Our thoughts and prayers are with Otto’s family and friends, and all who loved him SENTENCEEND SENTENCESTART   Full Statement: whitehouse SENTENCEEND SENTENCESTART gov/the-press-office/2017/06/19/statement-president-donald-j-trump-passing-otto-warmbier" Thank you Wyatt and Montana — two young Americans who aren’t afraid to stand up for what they believe in SENTENCE'




```python
from keras.preprocessing.text import text_to_word_sequence
text2 = text_to_word_sequence(text1, lower=False, split=" ") #using only 10000 first words
```


```python
text2[0:100]
```




    ['While',
     'I',
     'greatly',
     'appreciate',
     'the',
     'efforts',
     'of',
     'President',
     'Xi',
     'China',
     'to',
     'help',
     'w',
     'North',
     'Korea',
     'it',
     'has',
     'not',
     'worked',
     'out',
     'SENTENCEEND',
     'SENTENCESTART',
     'At',
     'least',
     'I',
     'know',
     'China',
     'tried',
     'The',
     'Democrats',
     'don’t',
     'want',
     'you',
     'to',
     'know',
     'how',
     'much',
     'we’ve',
     'accomplished',
     'SENTENCEEND',
     'SENTENCESTART',
     'While',
     'they',
     'do',
     'nothing',
     'but',
     'obstruct',
     'and',
     'distract',
     'we',
     'continue',
     'to',
     'work',
     'hard',
     'on',
     'fulfilling',
     'our',
     'promises',
     'to',
     'YOU',
     'Big',
     'day',
     'today',
     'SENTENCEEND',
     'SENTENCESTART',
     'VOTE',
     'Handel',
     'GA',
     '6',
     'and',
     'Norman',
     'SC',
     '5',
     'for',
     'Congress',
     'SENTENCEEND',
     'SENTENCESTART',
     'These',
     'two',
     'will',
     'be',
     'a',
     'tremendous',
     'help',
     'in',
     'our',
     'fight',
     'for',
     'lower',
     'taxes',
     'stronger',
     'security',
     'and',
     'great',
     'healthcare',
     'The',
     'U',
     'SENTENCEEND',
     'SENTENCESTART',
     'S']




```python
from keras.preprocessing.text import Tokenizer
token = Tokenizer(num_words=600,char_level=False)
token.fit_on_texts(text2)
```


```python
text_mtx = token.texts_to_matrix(text2, mode='binary')
```


```python
text_mtx.shape
```




    (2902, 600)




```python
input_ = text_mtx[:-1]
output_ = text_mtx[1:]
```


```python
input_.shape, output_.shape
```




    ((2901, 600), (2901, 600))




```python
#from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
```


```python
model = Sequential()
model.add(Embedding(input_dim=input_.shape[1],output_dim= 42, input_length=input_.shape[1]))
# the model will take as input an integer matrix of size (batch, vocabulary_size).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, vocabulary_size, 42), where None is the batch dimension.
```


```python
model.add(Flatten())
model.add(Dense(output_.shape[1], activation='sigmoid'))
```


```python
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])
model.fit(input_, y=output_, batch_size=300, epochs=10, verbose=1, validation_split=0.2)
```

    Train on 2320 samples, validate on 581 samples
    Epoch 1/10
    2320/2320 [==============================] - 5s - loss: 5.7183 - acc: 0.0112 - val_loss: 4.2990 - val_acc: 0.0086
    Epoch 2/10
    2320/2320 [==============================] - 4s - loss: 5.2992 - acc: 0.0362 - val_loss: 4.2388 - val_acc: 0.0499
    Epoch 3/10
    2320/2320 [==============================] - 4s - loss: 5.2088 - acc: 0.0431 - val_loss: 4.2266 - val_acc: 0.0499
    Epoch 4/10
    2320/2320 [==============================] - 4s - loss: 5.1529 - acc: 0.0431 - val_loss: 4.2374 - val_acc: 0.0499
    Epoch 5/10
    2320/2320 [==============================] - 5s - loss: 5.1151 - acc: 0.0431 - val_loss: 4.1933 - val_acc: 0.0499
    Epoch 6/10
    2320/2320 [==============================] - 4s - loss: 5.0782 - acc: 0.0431 - val_loss: 4.2048 - val_acc: 0.0499
    Epoch 7/10
    2320/2320 [==============================] - 6s - loss: 5.0582 - acc: 0.0431 - val_loss: 4.2005 - val_acc: 0.0499
    Epoch 8/10
    2320/2320 [==============================] - 6s - loss: 5.0229 - acc: 0.0431 - val_loss: 4.1965 - val_acc: 0.0499
    Epoch 9/10
    2320/2320 [==============================] - 6s - loss: 4.9949 - acc: 0.0431 - val_loss: 4.2167 - val_acc: 0.0499
    Epoch 10/10
    2320/2320 [==============================] - 6s - loss: 4.9741 - acc: 0.0513 - val_loss: 4.2140 - val_acc: 0.0551
    




    <keras.callbacks.History at 0x18210efa860>




```python
import numpy as np
def get_next(text,token,model,fullmtx,fullText):
    tmp = text_to_word_sequence(text, lower=False, split=" ")
    tmp = token.texts_to_matrix(tmp, mode='binary')
    p = model.predict(tmp)
    best10 = p.argsort() [0][-10:]
    bestMatch = np.random.choice(best10,1)[0]
    next_idx = np.min(np.where(fullmtx[:,bestMatch]>0))
    return fullText[next_idx]
```


```python
import random
text3 = ''
for x in range(0, 30):
    currWord = random.choice(text2)
    while currWord == unknown_token or currWord == sentence_start_token or currWord == sentence_end_token or currWord == separator:
        currWord = random.choice(text2)
    text3 = text3 + currWord
    for y in range(0, 20):
        nextWord = get_next(currWord,token,model,text_mtx,text2)
        if nextWord == sentence_start_token or nextWord == sentence_end_token:
            text3 = text3 + ' ' + sentence_end_token + ' ' + sentence_start_token + ' '
        else:
            text3 = text3 + ' ' + nextWord
        currWord = nextWord
        if y == 19:
            text3 = text3 + '.\n'
text3
```




    "OUR to we and our and we will we of we will SENTENCEEND SENTENCESTART  and we we our the SENTENCEEND SENTENCESTART  we a.\nTrump's of to to of SENTENCEEND SENTENCESTART  the our to and and our we of the and President Donald in the our.\nRoad and SENTENCEEND SENTENCESTART  we the SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  the our SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  to in to SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  the the to in SENTENCEEND SENTENCESTART .\nGermany to a to a the SENTENCEEND SENTENCESTART  the we will to the President we of to of and President we and.\ndedication we SENTENCEEND SENTENCESTART  and the and and a to the and SENTENCEEND SENTENCESTART  the we will we of and to SENTENCEEND SENTENCESTART  to.\nWorkforce of SENTENCEEND SENTENCESTART  President the President of our the President a and of we of and to our we a to.\nto of and we the and of our a a of the to to SENTENCEEND SENTENCESTART  a the SENTENCEEND SENTENCESTART  our SENTENCEEND SENTENCESTART  a.\nof to of SENTENCEEND SENTENCESTART  our SENTENCEEND SENTENCESTART  and SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  the President Donald to SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  to SENTENCEEND SENTENCESTART  the the the our.\nshooting SENTENCEEND SENTENCESTART  and our a in of President SENTENCEEND SENTENCESTART  of SENTENCEEND SENTENCESTART  to in the a to SENTENCEEND SENTENCESTART  of a and we.\nday SENTENCEEND SENTENCESTART  our SENTENCEEND SENTENCESTART  and to the the President SENTENCEEND SENTENCESTART  and of of the President SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  we of of we.\nwill in in SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  the our we and SENTENCEEND SENTENCESTART  we the SENTENCEEND SENTENCESTART  the of to we to SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART .\nthe SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  of our and the SENTENCEEND SENTENCESTART  of President of the President a to and SENTENCEEND SENTENCESTART  a a SENTENCEEND SENTENCESTART  we.\n6 a SENTENCEEND SENTENCESTART  to in a the President SENTENCEEND SENTENCESTART  in a SENTENCEEND SENTENCESTART  and our in SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  a and the to.\npaid SENTENCEEND SENTENCESTART  we will SENTENCEEND SENTENCESTART  and SENTENCEEND SENTENCESTART  our our SENTENCEEND SENTENCESTART  of our the SENTENCEEND SENTENCESTART  the the President Donald a a our.\nof of to we our we of of the of SENTENCEEND SENTENCESTART  President we SENTENCEEND SENTENCESTART  a of a SENTENCEEND SENTENCESTART  in to our.\nAmerica we SENTENCEEND SENTENCESTART  the SENTENCEEND SENTENCESTART  we to in the President our the of the and the a of of SENTENCEEND SENTENCESTART  we.\nthe SENTENCEEND SENTENCESTART  of our a of we of to our to of we we of we SENTENCEEND SENTENCESTART  to our a SENTENCEEND SENTENCESTART .\nand and our our to SENTENCEEND SENTENCESTART  of the SENTENCEEND SENTENCESTART  and and a the and to a our the we to of.\ncountry of we we will our to our the the President SENTENCEEND SENTENCESTART  we we to the of our we will and.\nReceives and the SENTENCEEND SENTENCESTART  to to and SENTENCEEND SENTENCESTART  our and our SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  and SENTENCEEND SENTENCESTART  a our SENTENCEEND SENTENCESTART  to in the.\nwill we to to and SENTENCEEND SENTENCESTART  President we the we SENTENCEEND SENTENCESTART  to and a the to our to we the to.\nNCAA we a SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  of SENTENCEEND SENTENCESTART  and and of SENTENCEEND SENTENCESTART  the our in we our of to the SENTENCEEND SENTENCESTART  of.\nyou SENTENCEEND SENTENCESTART  we our we we SENTENCEEND SENTENCESTART  in in SENTENCEEND SENTENCESTART  a the we to we the we will to SENTENCEEND SENTENCESTART  and.\nsure a to and to of the SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  a in to SENTENCEEND SENTENCESTART  we and a and the to the President.\nthe to a our our our the our and President SENTENCEEND SENTENCESTART  a a of SENTENCEEND SENTENCESTART  to SENTENCEEND SENTENCESTART  a SENTENCEEND SENTENCESTART  and SENTENCEEND SENTENCESTART .\nCongress and and and a SENTENCEEND SENTENCESTART  a to in the and and the and SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  our a SENTENCEEND SENTENCESTART  in.\nMike in we to to a we our the SENTENCEEND SENTENCESTART  our our our we of our we to SENTENCEEND SENTENCESTART  a and.\ndoing and SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  in SENTENCEEND SENTENCESTART  we and SENTENCEEND SENTENCESTART  we SENTENCEEND SENTENCESTART  a the the we of a to of to our.\nRegulations in and SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  to of of we and to we the SENTENCEEND SENTENCESTART  in the a of our to SENTENCEEND SENTENCESTART .\nform of we will the and we of to in we SENTENCEEND SENTENCESTART  SENTENCEEND SENTENCESTART  of and a a of SENTENCEEND SENTENCESTART  of President.\n"




```python
text3 = text3.replace(' '+ separator + ' ','--')
text3 = text3.replace(' ' +sentence_end_token +' '+ sentence_start_token + ' ', '.')
text3
```




    "OUR to we and our and we will we of we will. and we we our the. we a.\nTrump's of to to of. the our to and and our we of the and President Donald in the our.\nRoad and. we the.. the our.. to in to.. the the to in..\nGermany to a to a the. the we will to the President we of to of and President we and.\ndedication we. and the and and a to the and. the we will we of and to. to.\nWorkforce of. President the President of our the President a and of we of and to our we a to.\nto of and we the and of our a a of the to to. a the. our. a.\nof to of. our. and.. the President Donald to.. to. the the the our.\nshooting. and our a in of President. of. to in the a to. of a and we.\nday. our. and to the the President. and of of the President.. we of of we.\nwill in in.. the our we and. we the. the of to we to....\nthe.. of our and the. of President of the President a to and. a a. we.\n6 a. to in a the President. in a. and our in.. a and the to.\npaid. we will. and. our our. of our the. the the President Donald a a our.\nof of to we our we of of the of. President we. a of a. in to our.\nAmerica we. the. we to in the President our the of the and the a of of. we.\nthe. of our a of we of to our to of we we of we. to our a..\nand and our our to. of the. and and a the and to a our the we to of.\ncountry of we we will our to our the the President. we we to the of our we will and.\nReceives and the. to to and. our and our.. and. a our. to in the.\nwill we to to and. President we the we. to and a the to our to we the to.\nNCAA we a.. of. and and of. the our in we our of to the. of.\nyou. we our we we. in in. a the we to we the we will to. and.\nsure a to and to of the.. a in to. we and a and the to the President.\nthe to a our our our the our and President. a a of. to. a. and..\nCongress and and and a. a to in the and and the and... our a. in.\nMike in we to to a we our the. our our our we of our we to. a and.\ndoing and.. in. we and. we. a the the we of a to of to our.\nRegulations in and.. to of of we and to we the. in the a of our to..\nform of we will the and we of to in we.. of and a a of. of President.\n"




```python
new_path = 'C:/src/Academy/final/FinalProject/data/trump-output-txt.txt'
new_file = open(new_path,'w')
new_file.write(text3)
new_file.close()
```


```python

```
