# %%

import argparse
import os
import librosa
import numpy as np
import time as time
import librosa.output as lo
import tensorflow as tf
print(tf.VERSION)

parser = argparse.ArgumentParser()

# Example usage
# python infer.py -c train-techno\model.ckpt-72825 -ig train-techno\infer\infer.meta -n 100 --human_friendly True
# 'train-techno\model.ckpt-72825'
parser.add_argument('--checkpoint_name', '-c', type=str, required=True)

# 'train-techno\infer\infer.meta'
parser.add_argument('--infer_graph', '-ig', type=str, required=True)
parser.add_argument('--samples', '-n', type=str, default=100)
parser.add_argument('--human_friendly', type=bool, default=False)

args = parser.parse_args()

tf.reset_default_graph()
saver = tf.train.import_meta_graph(args.infer_graph)
graph = tf.get_default_graph()
sess = tf.InteractiveSession()

try:
    saver.restore(sess, args.checkpoint_name)

    # Sample latent vectors
    _z = (np.random.rand(args.samples, 100) * 2.) - 1.

    # Generate
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]

    start = time.time()
    _G_z = sess.run([G_z], {z: _z})
    print('Finished! (Took {} seconds)'.format(time.time() - start))

    os.makedirs(f'infer/{args.checkpoint_name}/', exist_ok=True)
    

    for i in range(args.samples):
        if args.human_friendly:
            raw_sample = np.stack([_G_z[0][i], _G_z[0][i], _G_z[0][i], _G_z[0][i]]).flatten()
            audio = librosa.effects.time_stretch(raw_sample, 130 / 120)
        else:
            audio = _G_z[0][i]
        lo.write_wav(f'infer/{args.checkpoint_name}/{i}.wav', audio, 16000, norm=False)
finally:
    sess.close()


# %%
