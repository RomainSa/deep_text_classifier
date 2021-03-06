import re
import numpy as np
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
import tensorflow as tf

# load 20 newsgroup dataset (11k documents)
data = fetch_20newsgroups(subset='train')

# sample data:
print(data.target_names)
print(data.filenames.shape)
print(data.data[0])
print(data.target.shape)
print(data.target[0])
print(data.target_names[data.target[0]])


def preprocess(s, lowercase=True, trim=True, trim_specials=True, allowed_chars=[], forbidden_chars=[u''],
               prefix='#', suffix='#', stopwords_list=stopwords.words('english')):
    """
    String data preprocessing method
    """
    if lowercase:
        s = s.lower()
    if trim:   # remove spaces and newlines
        s = re.sub('[\s+]', ' ', s)
    if trim_specials:   # remove special characters
        s = ''.join(char for char in s if ((char.isalnum() or char in allowed_chars or char == ' ') and \
                    char not in forbidden_chars))
    if len(prefix) > 0 or len(suffix) > 0 or len(stopwords) > 0 or len(forbidden_chars) > 0:
        s = ' '.join([prefix + word + suffix for word in s.split(' ') if word not in forbidden_chars and \
                      word not in stopwords_list])
    return s


data.data = [preprocess(s) for s in data.data]

# turns data into character 3-grams vectors
vectorizer = CountVectorizer(input='content', analyzer='char', ngram_range=(3, 3))
vectors = vectorizer.fit_transform(data.data)

# splits data between train and test
X_train, X_test, y_train, y_test = train_test_split(vectors, data.target, test_size=0.20, random_state=42)

# builds a Neural Network classifier
n1 = 300
n2 = 300
n3 = 128
lr = 0.001
batch_size = 64
n_epochs = 60


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


x_ = tf.placeholder("float", shape=[None, X_train.shape[1]])
y_ = tf.placeholder("float", shape=[None, len(data.target_names)])

keep_prob = tf.placeholder("float")

W1 = weight_variable([X_train.shape[1], n1])
b1 = bias_variable([n1])

W2 = weight_variable([n1, n2])
b2 = bias_variable([n2])

W3 = weight_variable([n2, n3])
b3 = bias_variable([n3])

W4 = weight_variable([n3, 20])
b4 = bias_variable([20])

h1 = tf.nn.relu(tf.matmul(x_, W1) + b1)
h1_drop = tf.nn.dropout(h1, keep_prob)

h2 = tf.nn.relu(tf.matmul(h1_drop, W2) + b2)
h2_drop = tf.nn.dropout(h2, keep_prob)

h3 = tf.nn.relu(tf.matmul(h2_drop, W3) + b3)
h3_drop = tf.nn.dropout(h3, keep_prob)

y = tf.nn.softmax(tf.matmul(h3_drop, W4) + b4)

cross_entropy = -tf.reduce_sum(y_*tf.log(y+1e-9))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(n_epochs * X_train.shape[0] // batch_size + 1):
        rand_idx = np.random.randint(low=0, high=X_train.shape[0], size=batch_size)
        batch = np.array(X_train[rand_idx, :].todense()),\
            label_binarize(y_train[rand_idx], classes=np.arange(len(data.target_names)))
        if i % 100 == 0:
            print('Step', i, 'of', n_epochs * X_train.shape[0] // batch_size + 1)
            print('Train error:', 1-accuracy.eval(feed_dict={x_: batch[0], y_: batch[1], keep_prob: 1.0}))
        train_step.run(feed_dict={x_: batch[0], y_: batch[1], keep_prob: 0.5})

    print('Test error:')
    print(1-accuracy.eval(feed_dict={x_: np.array(X_test.todense()),
                                     y_: label_binarize(y_test, classes=np.arange(len(data.target_names))),
                                     keep_prob: 1.0}))
