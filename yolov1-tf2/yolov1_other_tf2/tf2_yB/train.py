import sys,time
import net,loss,tfrecord_read
import tensorflow as tf
from tensorflow.keras import optimizers
BS = 5
optimizers=optimizers.Adam(lr=0.01)

for epoch in range(1,3):
    db_iter = tfrecord_read.train_gen(tfrecord_read.raw_datasets, batch_size=BS, random=True)

    total_steps = 17125 // BS
    if (17125 % BS) != 0:
        total_steps = total_steps +1

    start = time.perf_counter()
    for step in range(1,total_steps+1):
        rate = (step / total_steps) * 100
        star = '*' * int(step / total_steps * 50)
        dot = "." * (50 - int(step / total_steps * 50))
        dur = time.perf_counter() - start
        print("\r{:^4}/{} {:^3.0f}% [{}->{}] {:.2f}s".format(step,total_steps,rate,star, dot, dur), end="")

        batch_data = next(db_iter)
        with tf.GradientTape() as tape:
            out = net.yolo_net(tf.cast(batch_data[0],dtype=tf.float32))
            loss_ = loss.loss_fun(out,tf.cast(batch_data[1],dtype=tf.float32))
        grads = tape.gradient(loss_, net.yolo_net.trainable_variables)
        optimizers.apply_gradients(zip(grads,net.yolo_net.trainable_variables))
    print()
