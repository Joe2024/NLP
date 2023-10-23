class Category:
  positive = "positive"
  negative = "negative"

train_x = ["i love the experience", "this is a great customer service", "i like the phone support", "that was very helpfull", "the support is bad", "i hate the waiting time", "worst experience ever", "i am vert angry after chat with customer service"]
train_y = [Category.positive, Category.positive, Category.positive, Category.positive, Category.negative, Category.negative, Category.negative, Category.negative]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True)
train_x_vectors = vectorizer.fit_transform(train_x)

print(vectorizer.get_feature_names_out())
print(train_x_vectors.toarray())

from sklearn import svm

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

test_x = vectorizer.transform(['i hate the support'])

clf_svm.predict(test_x)