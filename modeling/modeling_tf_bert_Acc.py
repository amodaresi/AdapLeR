import tensorflow as tf
from transformers import TFBertForSequenceClassification, GradientAccumulator

class TFBertForSequenceClassificationWAcc(TFBertForSequenceClassification):
  def compile(self, optimizer, loss, accum_steps=1, class_weights=None):
    super(TFBertForSequenceClassificationWAcc, self).compile(optimizer=optimizer, loss=loss)
    self.gradient_accumulator = GradientAccumulator()
    self._ACCUM_STEPS = accum_steps
    self.class_weights = class_weights

  @tf.function
  def apply_gradients(self, gradients=None):
    if self._ACCUM_STEPS > 1:
      self.optimizer.apply_gradients(zip(self.gradient_accumulator.gradients, self.trainable_weights))
      self.gradient_accumulator.reset()
    else:
      self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
    return None

  @tf.function
  def dummy_apply(self, gradients=None):
    return None

  @tf.function
  def train_step(self, inputs):
      x = inputs[0]
      y = inputs[1]
      grads, loss_value = self.compute_grads(x, y)
      if self._ACCUM_STEPS > 1:
        self.gradient_accumulator(grads)
        tf.cond(self.gradient_accumulator.step == self._ACCUM_STEPS, self.apply_gradients, self.dummy_apply)
      else:
        self.apply_gradients(grads)
      return {"loss": loss_value}
  
  @tf.function
  def compute_grads(self, x, y):
    with tf.GradientTape() as tape:
      logits = self(x, training=True)
      loss_value = self.compiled_loss(y, logits, sample_weight=None if self.class_weights is None else tf.gather(self.class_weights, y))
      divided_loss = loss_value / tf.cast(self._ACCUM_STEPS, tf.float32)
    grads = tape.gradient(divided_loss, self.trainable_weights)
    return grads, loss_value