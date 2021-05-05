# %%

# import tensorflow as tf
# import numpy as np
# import librosa
# from IPython.display import display, Audio
# import os

# # Load the graph
# tf.reset_default_graph()
# saver = tf.train.import_meta_graph('train-techno\model.ckpt-21002.meta')
# graph = tf.get_default_graph()
# sess = tf.InteractiveSession()
# saver.restore(sess, 'train-techno\model.ckpt-21002')

# # Create 50 random latent vectors z
# _z = (np.random.rand(64, 100) * 2.) - 1

# # Synthesize G(z)
# z = graph.get_tensor_by_name('random_uniform:0')
# G_z = graph.get_tensor_by_name('G_z:0')
# _G_z = sess.run(G_z, {z: _z})

# # Play audio in notebook
# # display(Audio(_G_z[0, :, 0], rate=16000))

# os.makedirs('infer', exist_ok=True)
# # for i in range(64):
# display(Audio(_G_z, rate=16000))
# # x = np.array(_G_z)
# # librosa.output.write_wav(f'infer/sample-{1}.wav', x, 16000)

# # import tensorflow as tf
# # tf.reset_default_graph()

# # saver = tf.train.import_meta_graph('infer.meta')
# # graph = tf.get_default_graph()
# # sess = tf.InteractiveSession()
# # saver.restore(sess, 'model.ckpt-10000')

# # z_n = graph.get_tensor_by_name('samp_z_n:0')
# # _z = sess.run(graph.get_tensor_by_name('samp_z:0'), {z_n: 10})

# # z = graph.get_tensor_by_name('G_z:0')
# # _G_z = sess.run(graph.get_tensor_by_name('G_z:0'), {z: _z})
# %%
import tensorflow as tf
tf.reset_default_graph()

saver = tf.train.import_meta_graph('train-techno\infer\infer.meta')
graph = tf.get_default_graph()
sess = tf.InteractiveSession()
saver.restore(sess, 'train-techno\model.ckpt-21002')

z_n = graph.get_tensor_by_name('samp_z_n:0')
_z = sess.run(graph.get_tensor_by_name('samp_z:0'), {z_n: 10})

z = graph.get_tensor_by_name('G_z:0')
_G_z = sess.run(graph.get_tensor_by_name('G_z:0'), {z: _z})
print('')