documents=['DataSet/test'+str(i)+'.txt' for i in range(1,76)]
# print(docuemnts)

output_file=open('results.txt', 'w')


from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import NMF


def parsing_files():
  global documents
  for doc in documents:

    file_to_read=open(doc, 'r')
    Docs=[]
    Docs.append(file_to_read.read())


    vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)

    input_matrix = vectorizer.fit_transform(Docs).todense()

    # print(input_matrix)

    # svd_modeling = TruncatedSVD(n_components=4, algorithm='randomized', n_iter=100, random_state=122)
    #
    # svd_modeling.fit(input_matrix)
    #
    # components = svd_modeling.components_

    NMF_model = NMF(n_components=4, random_state=1)
    W = NMF_model.fit_transform(input_matrix)
    H = NMF_model.components_

    vocab = vectorizer.get_feature_names_out()

    topic_word_list = []

    def get_topics(H):

      for i, comp in enumerate(H):
        terms_comp = zip(vocab, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:9]
        topic = " "
        for t in sorted_terms:
          topic = topic + ' ' + t[0]
        topic_word_list.append(topic)
      # print(topic_word_list)
      for t in topic_word_list:
        output_file.write(str(t)+'  ')
      output_file.write('\n')
      return topic_word_list

    get_topics(H)

parsing_files()
