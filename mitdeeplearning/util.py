import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np

from IPython import display as ipythondisplay
from string import Formatter


def get_tune(song):
    song = song.lstrip('\n')
    lines = song.split('\n')
    tunelines = lines[5:]
    return '\n'.join(tunelines)

def generate_song_with_key_tune(tune, key="nan"):
    '''
    Returns a string for a complete song with key and tune information
    :param key: a string e.g. 'A Major'
    :param tune: a list of lines of the new song
    :return: a complete song that is ready to play
    '''
    if type(tune) is str:
        if tune[0] == 'K':
            parts = tune.split('\n')
            if key == "nan":
                key = parts[0][2:]
            tunejoined = '\n'.join(parts[1:])
        else:
            tunejoined = tune

    else:
        tunejoined = '\n'.join(tune)

    header = 'X:1\nT:Name\nZ: id:dc-hornpipe-1\nM:C|\nL:1/8'
    song = f"{header}\nK:{key}\n{tunejoined}"
    return song


def display_model(model):
  tf.keras.utils.plot_model(model,
             to_file='tmp.png',
             show_shapes=True)
  return ipythondisplay.Image('tmp.png')


def plot_sample(x,y,vae):
    plt.figure(figsize=(2,1))
    plt.subplot(1, 2, 1)

    idx = np.where(y==1)[0][0]
    plt.imshow(x[idx])
    plt.grid(False)

    plt.subplot(1, 2, 2)
    _, _, _, recon = vae(x)
    recon = np.clip(recon, 0, 1)
    plt.imshow(recon[idx])
    plt.grid(False)

    plt.show()



class LossHistory:
  def __init__(self, smoothing_factor=0.0):
    self.alpha = smoothing_factor
    self.loss = []
  def append(self, value):
    self.loss.append( self.alpha*self.loss[-1] + (1-self.alpha)*value if len(self.loss)>0 else value )
  def get(self):
    return self.loss

class PeriodicPlotter:
  def __init__(self, sec, xlabel='', ylabel='', scale=None):

    self.xlabel = xlabel
    self.ylabel = ylabel
    self.sec = sec
    self.scale = scale

    self.tic = time.time()

  def plot(self, data):
    if time.time() - self.tic > self.sec:
      plt.cla()

      if self.scale is None:
        plt.plot(data)
      elif self.scale == 'semilogx':
        plt.semilogx(data)
      elif self.scale == 'semilogy':
        plt.semilogy(data)
      elif self.scale == 'loglog':
        plt.loglog(data)
      else:
        raise ValueError("unrecognized parameter scale {}".format(self.scale))

      plt.xlabel(self.xlabel); plt.ylabel(self.ylabel)
      ipythondisplay.clear_output(wait=True)
      ipythondisplay.display(plt.gcf())

      self.tic = time.time()
