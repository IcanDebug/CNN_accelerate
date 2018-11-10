import tensorflow as tf
from mobilenet_v2 import mobilenetv2
import input_data

import time
import os


height=224
width=224

learning_rate = 0.00001

sess=tf.Session()

# read queue, read img data
#glob_pattern = os.path.join(args.dataset_dir, '*.tfrecord')
#tfrecords_list = glob.glob(glob_pattern)
#filename_queue = tf.train.string_input_producer(tfrecords_list, num_epochs=None)
#img_batch, label_batch = get_batch(filename_queue, args.batch_size)
train_dir = '/home/pdd/pdwork/CV_BiShe/picture/picTest/'

train, train_label = input_data.get_files(train_dir)

img_batch, label_batch = input_data.get_batch(train,train_label,width,height,100, 10000)
##
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#inputs = tf.placeholder(tf.float32, [None, height, width, 3], name='input')
logits, pred=mobilenetv2(img_batch, num_classes=2, is_train=True)

# loss
loss_ = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=logits))
# L2 regularization
l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
total_loss = loss_ + l2_loss

# evaluate model, for classification
correct_pred = tf.equal(tf.argmax(pred, 1), tf.cast(label_batch, tf.int64))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# learning rate decay
base_lr = tf.constant(learning_rate)
lr_decay_step = 20000 // 100 * 2  # every epoch
global_step = tf.placeholder(dtype=tf.float32, shape=())
lr = tf.train.exponential_decay(base_lr, global_step=global_step, decay_steps=lr_decay_step,
                                decay_rate=0.96)
# optimizer
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9, momentum=0.9)
    train_op = tf.train.AdamOptimizer(
        learning_rate=lr).minimize(total_loss)

# summary
tf.summary.scalar('total_loss', total_loss)
tf.summary.scalar('accuracy', acc)
tf.summary.scalar('learning_rate', lr)
summary_op = tf.summary.merge_all()

# summary writer
writer = tf.summary.FileWriter('log/', sess.graph)

sess.run(tf.global_variables_initializer())

# saver for save/restore model
saver = tf.train.Saver()
# load pretrained model
step=0
#if not args.renew:
#    print('[*] Try to load trained model...')
#    could_load, step = load(sess, saver, args.checkpoint_dir)

max_steps = int(20000 / 100 * 20)

print('START TRAINING...')
for _step in range(step+1, max_steps+1):
    start_time=time.time()
    feed_dict = {global_step:_step}#, inputs:img_batch}
    # train
    _, _lr = sess.run([train_op, lr], feed_dict=feed_dict)
    # print logs and write summary
    if _step % 10 == 0:
        _summ, _loss, _acc = sess.run([summary_op, total_loss, acc],
                                   feed_dict=feed_dict)
        writer.add_summary(_summ, _step)
        print('global_step:{0}, time:{1:.3f}, lr:{2:.8f}, acc:{3:.6f}, loss:{4:.6f}'.format
              (_step, time.time() - start_time, _lr, _acc, _loss))

    # save model
    if _step % 10 == 0:
        save_path = saver.save(sess, os.path.join('log/', 'mobileV2'), global_step=_step)
        print('Current model saved in ' + save_path)

tf.train.write_graph(sess.graph_def, 'log/', 'mobileV2' + '.pb')
save_path = saver.save(sess, os.path.join('log/', 'mobileV2'), global_step=max_steps)
print('Final model saved in ' + save_path)
sess.close()
print('FINISHED TRAINING.')

