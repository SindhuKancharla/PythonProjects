
# coding: utf-8

# In[222]:

from __future__ import division
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as ss
import os
import re
import traceback
from happierfuntokenizing import Tokenizer

import math
import operator

users = {}


# ### 1) Method for reading all files, parsing the blogs and tokenizing them

# In[223]:

def parse_blogs(path):
    
    tokenizer = Tokenizer()
    users = {}
    global_words_dict = {}
    industry_map = {}
    total_users = 0
    total_blog_posts = 0
    iterations = 0
    topics = pd.read_csv('wwbpFBtopics_condProb.csv')
    
    regex = r'<post>(.*?)</post>'

    for filename in os.listdir(path):
        iterations += 1
        print "user %d" %iterations
        if iterations > 50:
            break
            
        if filename.startswith("."):
            continue
            
        parts = filename.split(".")

        user_attributes_map = {}
        word_count_map = {}
        topic_prob_map = {}

        user_total_words_count = 0
        
        user_id = (int)(parts[0])
        gender = parts[1]
        
        if gender == "male":
            gender = 0
        else:
            gender = 1
            
        age = (int)(parts[2])
        industry = parts[3]
        star_sign = parts[4]
        
        if user_id in users:
            user_attributes_map = users[user_id]
        
        if industry in industry_map:
            industry_map[industry] = industry_map[industry] + 1
        else:
            industry_map[industry] = 1
                
        with open(path + filename, 'r') as user_blog:
            user_blogs = user_blog.read().replace('\n', '').replace('\r', '').replace('\t', '')
    
        all_blog_posts = re.findall(regex, user_blogs, re.DOTALL)

        total_blog_posts = total_blog_posts + len(all_blog_posts)

        for blog in all_blog_posts:  
            words = tokenizer.tokenize(blog.strip())
            user_total_words_count = user_total_words_count + len(words)
            
            if 'wc_map' in user_attributes_map:
                word_count_map = user_attributes_map['wc_map']

            for word in words:
                if word in word_count_map:
                    count = word_count_map[word]
                    count = count + 1
                    word_count_map[word] = count
                else:
                    word_count_map[word] = 1
                    
                if word in global_words_dict:
                    count = global_words_dict[word]
                    count = count + 1
                    global_words_dict[word] = count
                else:
                    global_words_dict[word] = 1
                    

        for topic in range(2000):
            prob_topic_given_user = 0.0

            topic_dict = topics[topics['category'] == topic]

            for row in topic_dict.itertuples():
                word = row[1]
                prob_topic_given_word = row[3]
                if word in word_count_map:
                    count_user_word = word_count_map[word]
                    prob_word_given_user = count_user_word/user_total_words_count

                    cur = prob_topic_given_word * prob_word_given_user

                    prob_topic_given_user = prob_topic_given_user + cur
            
            topic_prob_map[topic] = prob_topic_given_user
        
        user_attributes_map['wc_map'] = word_count_map
        user_attributes_map['age'] = age
        user_attributes_map['industry'] = industry
        user_attributes_map['star_sign'] = star_sign
        user_attributes_map['user_id'] = user_id
        user_attributes_map['topic_prob_map'] = topic_prob_map
        user_attributes_map['total_count'] = user_total_words_count
        user_attributes_map['gender'] = gender
        
        users[user_id] = user_attributes_map
    return (users, global_words_dict, industry_map, total_blog_posts)


# In[224]:

if __name__ == '__main__':
    
    
    global users
    
    path = "blogs/"
    users, all_words_counts, industry_users_map, num_blogs  = parse_blogs(path)
    
    print "1.a) posts: %d " % num_blogs
    print "1.b) users: %d " % len(users)
    print "1.c) words: %d " % len(all_words_counts)
    print "1.d) "
    print industry_users_map
    
    #read_all_blogs(users_words_counts,users_topics_probs)
    


# ### 2) Calculating first 3 users' probability of mentioning 3 given topics

# In[225]:

topicwordsfile = pd.read_csv("wwbpFBtopics_condProb.csv")

topic_map = {}

topic_map[463] = topicwordsfile[topicwordsfile["category"]==463]
topic_map[963] = topicwordsfile[topicwordsfile["category"]==963]
topic_map[981] = topicwordsfile[topicwordsfile["category"]==981]

user_ids = sorted(k for k in users)
lowest_user_ids = sorted(user_ids)[0:3]

print "2.a)"
for userid in lowest_user_ids:
    print "%d" %userid
    
    for topic in topic_map:
        print str(topic) + " : "
        
        prob_topic_given_user = users[userid]["topic_prob_map"][topic]
        print prob_topic_given_user
        


# ### 3) Correlate each topic usage with user age, adjusting for gender

# In[226]:

def multiply_matrices(Right,Left):
    
    Right = np.matrix(Right)
    Left = np.matrix(Left)

    Betas = ((((Right.T)*Right).I)*(Right.T))*Left
    return Betas
    


# In[227]:


# T is topic usage , A is User's Age , G is User's Gender

alpha = 0.05
#### Bonferroni Correction 
Topic_userprob_map = {}

alpha = alpha/2000
Beta_topics = {}
Pvalues_topics = {}
Sig_topics = {}

for topic in range(2000):
    
    Topics = []
    Ages = []
    Genders = []
    for user_id,user_attributes_map in users.iteritems():
                   
        if topic in Topic_userprob_map:
            tmp_list = Topic_userprob_map[topic]
        else:
            tmp_list = []
           
        tmp_list.append((user_id,user_attributes_map["topic_prob_map"][topic]))
        Topic_userprob_map[topic] = tmp_list
        
        
        Ages.append(user_attributes_map["age"])
        Genders.append(user_attributes_map["gender"])
        Topics.append(user_attributes_map["topic_prob_map"][topic])       
        

    # STANDARDIZING
    Ages = (Ages - np.mean(Ages)) / np.std(Ages)
    Genders = (Genders - np.mean(Genders)) / np.std(Genders)
    Topics = (Topics - np.mean(Topics)) / np.std(Topics)

    X = []
    for i in range(len(Topics)):
        X.append([Topics[i], Genders[i]])
 
    Y = []
    for i in range(len(Ages)):
        Y.append([Ages[i]])
 
    Betas = multiply_matrices(X,Y)
    
    Beta_topic = Betas.item(0)
    Beta_gender = Betas.item(1)
    
    mean_topics = np.mean(Topics)
    var_topics = 0
    RSS = 0
    
    
    N = len(users)
    ### Calculating RSS
    for i in range(N):
        yi = Ages[i]
            
        xi = (Beta_topic * Topics[i]) + (Beta_gender * Genders[i])
        
        var_topics = var_topics + ((Topics[i]-mean_topics)**2)
        RSS = RSS + ((yi-xi)**2)
    
    s2 = RSS/(N-2-1)
        
    denom = math.sqrt(s2/var_topics)
    t = Beta_topic/denom
    pval = ss.t.sf(np.abs(t), N-1)*2
    
    if pval < alpha:
        Sig_topics[topic] = "Y"
    else:
        Sig_topics[topic] = "N"
        
    Pvalues_topics[topic] = pval
    Beta_topics[topic] = Beta_topic
    
    
sorted_pos = sorted(Beta_topics.items(), key=operator.itemgetter(1))[:10]
sorted_neg = sorted(Beta_topics.items(), key=operator.itemgetter(1), reverse= True)[:10]

print "3.a)"
for i in sorted_neg:
    print "topic_id : %d , correlation : %f , p-value : %f , signficant after correction? : %s"     % (i[0],i[1],Pvalues_topics[i[0]],Sig_topics[i[0]])


print "3.b)"
for i in sorted_pos:

    print "topic_id : %d , correlation : %f , p-value : %f , signficant after correction? : %s"     % (i[0],i[1],Pvalues_topics[i[0]],Sig_topics[i[0]])
    


# ### 4) Correlate each topic usage with user industry, adjusting for gender and age.

# In[233]:

industry_topic_cor_map = {}

Ones = [1]*len(users)
for industry, user_count in industry_users_map.iteritems():
    
    if user_count < 3:
        continue
        
    topic_corelation_map = {}
    
    for topic in range(2000):

        beta_0 = 0.0
        beta_topic= 0.0
        beta_age = 0.0
        beta_gender = 0.0
        
        Ages = []
        Genders = []
        Topics = []
        Industries = []

        for user_id, user_attributes_map in users.iteritems():

            Ages.append(user_attributes_map["age"])
            Genders.append(user_attributes_map["gender"])
            Topics.append(user_attributes_map["topic_prob_map"][topic])
            ind_name = user_attributes_map["industry"]

            Industries.append(1 if ind_name == industry else 0)

        # STANDARDIZING
        Ages = (Ages - np.mean(Ages)) / np.std(Ages)
        Genders = (Genders - np.mean(Genders)) / np.std(Genders)
        Topics = (Topics - np.mean(Topics)) / np.std(Topics)

        X = []
        for i in range(len(Topics)):
            X.append([Topics[i], Genders[i], Ages[i]])

        Y = []
        for i in range(len(Industries)):
            Y.append([Industries[i]])

        X = np.matrix(X)
        Y = np.matrix(Y)

        try:
            while(1):
                prev_beta_topic = beta_topic

                beta_matrix = np.matrix([beta_topic,beta_gender,beta_age,beta_0])
                diag = [0]*len(users)
                z = [0]*len(users)

                for i in range(len(users)):
                    q = math.exp(beta_0 + Topics[i]*beta_topic + Genders[i]*beta_gender + Ages[i]*beta_age)
                    p = q/(1+q)

                    diag[i] = p*(1-p)
                    z[i] = math.log(p/(1-p)) + ((Industries[i] - p)/(p*(1-p)))

                W = np.matrix(np.diag(diag))
                Z = np.matrix(z).T

                beta = (((X.T)*W*X).I)*(X.T)*W*(Z)
                beta_topic = beta.item(0)

                if (round(beta_topic,2) - round(prev_beta_topic,2) < 0.01):
                    break
        except Exception as e:
            #print e
            continue
            
        topic_corelation_map[topic] = beta_topic
    
    topic_corelation_map = sorted(topic_corelation_map.items(), key=operator.itemgetter(1))
    industry_topic_cor_map[industry] = topic_corelation_map


# In[234]:


all_top_topics = []

print "4)"
for industry,topic_cor_map in industry_topic_cor_map.iteritems():
    #print topic_cor_map
    top5pairs = topic_cor_map[-5:]
    bottom5pairs = topic_cor_map[:5]

    #print top5pairs
    print "a)"
    for tid in top5pairs:
        
        topic = tid[0]
        if topic not in all_top_topics:
            all_top_topics.append(topic)
            
        print "Industry : %s , topic_id : %d , coefficient : %.3f "% (industry,tid[0],tid[1])
    
    print "b)"
    for tid in bottom5pairs:
        
        topic = tid[0]
        if topic not in all_top_topics:
            all_top_topics.append(topic)
        print "Industry : %s , topic_id : %d , coefficient : %.3f "% (industry,tid[0],tid[1])
    #break


# ### 5) Plot topics by industry x age.

# In[1]:



topics_terms_pd = pd.read_csv("2000topics.top20freqs.keys.csv")

Topic_ages_mean = {}

for topic in all_top_topics:
    
    tps = Topic_userprob_map[topic]

    tps.sort(key=operator.itemgetter(1),reverse=True)
    twnty_five_percent = int(round(len(tps)/4))
        
    tfpercent = tps[:twnty_five_percent]

    ages = []
    for ut in tfpercent:
        uid = ut[0]
        ages.append(users[uid]['age'])

    ages_mean = np.mean(ages)
    Topic_ages_mean[topic] = round(ages_mean,3)
    

N = len(industry_topic_cor_map)  
c = 0

fig = plt.figure(figsize=(15,15))
fig.subplots_adjust(top=7,bottom=1,right=5)
fig.suptitle("Topics By Industry", fontsize=20, fontweight='bold')

for industry,topic_cor_map in industry_topic_cor_map.iteritems():

    c += 1
    top5pairs = topic_cor_map[-5:]
    bottom5pairs = topic_cor_map[:5]

    top5pairs.extend(bottom5pairs)
    correlations = [tup[1] for tup in top5pairs]
        
    X = []
    Y = []
    topic_terms_map = {}
    for tid in top5pairs:
        
        words_list = []
        topic = tid[0]
        
        cor = tid[1]
        
        #cor = (cor-np.mean(correlations))/ np.std(correlations)
        X.append(round(cor,3))
        Y.append(Topic_ages_mean[topic])
        
        topic_terms = topics_terms_pd.iloc[topic-1]
    
        for wd in topic_terms.values:
            if isinstance(wd, str):
                if len(words_list) > 4:
                    break
                else:
                    words_list.append(wd)
                
        topic_terms_map[topic] = words_list
    
    tpc_words_list = topic_terms_map.values()


    ax = fig.add_subplot(N,1,c)
    ax.axis([np.min(X),np.max(X), np.min(Y), np.max(Y)])
    
    ax.set_title(industry,fontsize=35)
    ax.set_xlabel('Topic Correlation with Industry',fontsize=35)
    ax.set_ylabel('Mean Age of top 25% users',fontsize=35)
    
    for i in range(10):
        words = tpc_words_list[i]
        txt = ""
        for j in range(len(words)-1):
            txt = words[j] +" , " +words[j+1] + "\n" 
            j = j+1
        ax.text(X[i], Y[i], txt, fontsize=35)
    
    plt.show()

plt.savefig('5a_industry_plots.png')


# In[ ]:



